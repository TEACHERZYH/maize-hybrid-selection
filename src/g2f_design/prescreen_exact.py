"""E3的精确预筛实现：保留逐步剩余池定义与冻结平局规则。"""
import heapq
import numpy as np
from .tessar import TESSAR
from .design import best_index


def best_k(scores, identifiers, count):
    scores=np.asarray(scores,dtype=np.float64)
    if len(scores)!=len(identifiers) or not 0<=count<=len(scores):
        raise ValueError('预筛个数或ID不合法')
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise FloatingPointError('预筛分数非有限')
    order=np.argsort(-scores,kind='stable');removed=np.zeros(len(scores),dtype=bool)
    maximum_pointer=0;push_pointer=0;heap=[];chosen=[]
    for _ in range(count):
        while removed[order[maximum_pointer]]:maximum_pointer+=1
        maximum=scores[order[maximum_pointer]]
        threshold=-np.inf if maximum==-np.inf else maximum-1e-12*max(1.,abs(maximum))
        while push_pointer<len(order) and scores[order[push_pointer]]>=threshold:
            index=int(order[push_pointer]);heapq.heappush(heap,(identifiers[index],index));push_pointer+=1
        _,index=heapq.heappop(heap);removed[index]=True;chosen.append(index)
    return chosen


class E3TESSAR(TESSAR):
    def suggest(self, remaining):
        if not len(remaining):return []
        indices=np.asarray([self.local_index[int(i)] for i in remaining],dtype=int)
        q=min(256,len(indices))
        scores=self.log_scores(indices,q,self.fresh_l(indices))
        return [int(remaining[i]) for i in best_k(scores,[self.identifiers[i] for i in remaining],q)]

    def add(self, global_index):
        if global_index not in self.local_index or global_index in self.selected:
            raise ValueError('E3重复或不合法的真实选择')
        self.selected.append(global_index)
        self._add_anchors(self.all_points[[global_index]],[self.identifiers[global_index]])
        # E3每次suggest重新计算L；原版固定全池的累计最小L不参与该协议。


class ExactPrescreenGreedy:
    def __init__(self,posterior,observations,identifiers,initial,h,prescreen):
        self.posterior=posterior.copy();self.a=np.asarray(observations,dtype=np.float64)
        self.identifiers=list(identifiers);self.selected=list(initial);self.h=np.asarray(h,dtype=np.float64)
        self.prescreen=prescreen;self.legal=np.ones(len(self.a),dtype=bool)
        if len(set(identifiers))!=len(identifiers) or len(set(initial))!=len(initial):
            raise ValueError('候选或S0重复')
        self.legal[initial]=False
        for index in initial:self.posterior.condition_position(self.a[index])
        self.steps=0;self.refresh()

    def refresh(self):
        self.r=self.a@self.posterior.covariance
        self.rh=self.r@self.h
        self.den=np.einsum('ij,ij->i',self.r,self.a)+self.posterior.noise

    def take(self,budget):
        if not len(self.selected)<=budget<=len(self.a):
            raise ValueError('E3预算不合法')
        while len(self.selected)<budget:
            remaining=np.flatnonzero(self.legal).tolist();candidates=self.prescreen.suggest(remaining)
            if len(candidates)!=min(256,len(remaining)) or len(set(candidates))!=len(candidates) or not set(candidates).issubset(remaining):
                raise ValueError('E3预筛候选不完整或不合法')
            numerator=np.einsum('ij,ij->i',self.r[candidates],self.rh[candidates])
            if np.min(numerator)<-1e-10*max(1.,float(np.max(np.abs(numerator)))) or np.any(self.den[candidates]<=0):
                raise FloatingPointError('E3候选条件方差或H分数不合法')
            scores=np.maximum(numerator,0)/self.den[candidates]
            offset=best_index(scores,[self.identifiers[i] for i in candidates],np.ones(len(candidates),dtype=bool))
            index=int(candidates[offset]);row=self.a[index]
            r=self.posterior.covariance@row;s=float(row@r+self.posterior.noise);c=self.a@r
            self.rh-=np.outer(c,self.h@r)/s;self.r-=np.outer(c,r)/s;self.den-=c*c/s
            self.posterior.condition_position(row);self.selected.append(index);self.legal[index]=False
            self.prescreen.add(index);self.steps+=1
            if self.steps%256==0:self.refresh()
        self.refresh()
        return self.selected.copy()
