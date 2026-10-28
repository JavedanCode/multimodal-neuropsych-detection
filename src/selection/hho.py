"""Explicit sigmoid-transfer binary adaptation of continuous HHO equations.

References and variant choices are documented in docs/stage7_protocol.md.
Fitness sees binary masks only. A decoded mask is evaluated once per proposal;
its score remains paired with its latent position throughout acceptance decisions.
"""
from __future__ import annotations
import math
from typing import Callable
import numpy as np


def levy(rng: np.random.Generator, dimensions: int) -> np.ndarray:
    beta=1.5
    scale=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    return rng.normal(size=dimensions)*scale/np.maximum(np.abs(rng.normal(size=dimensions)),1e-12)**(1/beta)


def decode(position: np.ndarray, rng: np.random.Generator, groups: list[np.ndarray]) -> np.ndarray:
    probability=1/(1+np.exp(-np.clip(position,-6,6)))
    mask=rng.random(len(position))<probability
    for group in groups:
        if not mask[group].any():
            mask[group[np.argmax(position[group])]]=True
    return mask


def binary_hho(fitness: Callable[[np.ndarray],float], dimensions: int,
               groups: list[np.ndarray], seed: int=42, population: int=8, iterations: int=8) -> dict:
    if dimensions<2 or population<2 or iterations<1 or not groups:
        raise ValueError('Invalid optimizer dimensions or budget')
    flattened=np.concatenate(groups)
    if any(len(g)==0 for g in groups) or sorted(flattened.tolist())!=list(range(dimensions)):
        raise ValueError('Groups must partition all feature indices')
    rng=np.random.default_rng(seed)
    positions=rng.uniform(-6,6,(population,dimensions))
    masks=[decode(x,rng,groups) for x in positions]
    # Include the no-selection reference in the initial population.
    positions[0]=6.;masks[0]=np.ones(dimensions,dtype=bool)
    scores=np.array([fitness(mask.copy()) for mask in masks])
    if not np.isfinite(scores).all():raise ValueError('Fitness must be finite')
    best=int(np.argmin(scores));rabbit=positions[best].copy();best_mask=masks[best].copy();best_score=float(scores[best])
    history=[{'iteration':0,'best_fitness':best_score,'selected_features':int(best_mask.sum())}]
    branches={k:0 for k in ['explore_peer','explore_range','soft','hard','soft_dive','hard_dive']}
    calls=population
    def evaluate(position):
        nonlocal calls
        position=np.clip(position,-6,6)
        mask=decode(position,rng,groups)
        score=float(fitness(mask.copy()));calls+=1
        if not np.isfinite(score):raise ValueError('Fitness must be finite')
        return position,mask,score
    for t in range(iterations):
        # Synchronous population snapshot; document this choice vs in-place updates.
        old=positions.copy();mean=old.mean(axis=0)
        for i,x in enumerate(old):
            energy=2*(1-t/iterations)*(2*rng.random()-1)
            jump=2*(1-rng.random())
            if abs(energy)>=1:
                if rng.random()<.5:
                    branches['explore_peer']+=1
                    peer=old[rng.integers(population)]
                    proposal=peer-rng.random()*np.abs(peer-2*rng.random()*x)
                else:
                    branches['explore_range']+=1
                    proposal=(rabbit-mean)-rng.random()*(-6+12*rng.random())
                accepted=evaluate(proposal)
            elif rng.random()>=.5:
                if abs(energy)>=.5:
                    branches['soft']+=1
                    proposal=(rabbit-x)-energy*np.abs(jump*rabbit-x)
                else:
                    branches['hard']+=1
                    proposal=rabbit-energy*np.abs(rabbit-x)
                accepted=evaluate(proposal)
            else:
                soft=abs(energy)>=.5
                branches['soft_dive' if soft else 'hard_dive']+=1
                proposal=rabbit-energy*np.abs(jump*rabbit-(x if soft else mean))
                first=evaluate(proposal)
                if first[2]<scores[i]:
                    accepted=first
                else:
                    second=evaluate(proposal+rng.random(dimensions)*levy(rng,dimensions))
                    accepted=second if second[2]<scores[i] else (positions[i],masks[i],scores[i])
            positions[i],masks[i],scores[i]=accepted
            if scores[i]<best_score:
                best_score=float(scores[i]);rabbit=positions[i].copy();best_mask=masks[i].copy()
        history.append({'iteration':t+1,'best_fitness':best_score,'selected_features':int(best_mask.sum())})
    return {'mask':best_mask.tolist(),'fitness':best_score,'history':history,'branches':branches,'objective_calls':calls}
