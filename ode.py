# -*- coding: utf-8 -*-

from typing import AnyStr, Callable
from collections import defaultdict

import numpy as np

class OdeSolver:
    def __init__(self, tend:Callable, dt:float, scheme:AnyStr):
        self.tend = tend
        self.dt = dt
        self.scheme = getattr(self, scheme)

    @staticmethod
    def euler(state, dt, tend):
        while True:
            state = state + dt*tend(state)
            yield state

    @staticmethod
    def rk4(state, dt, tend):
        while True:
            k1 = dt*tend(state)
            k2 = dt*tend(state+k1/2)
            k3 = dt*tend(state+k2/2)
            k4 = dt*tend(state+k3)
            state = state + (k1+2*(k2+k3)+k4)/6
            yield state

    @staticmethod
    def leapfrog(state, dt, tend):
        prev, state = (state, state + dt*tend(state)) # euler
        while True:
            yield state
            prev, state = (state, prev + (2*dt)*tend(state))

    @staticmethod
    def ab2(state, dt, tend):
        tprev = dt*tend(state)
        prev, state = (state, state + tprev) # euler
        tnow = dt*tend(state)
        while True:
            prev, state = (state, state + 1.5*tnow -0.5*tprev)
            yield state
            tprev, tnow = (tnow, dt*tend(state))

    def iter_states(self, state:np.ndarray):
        return self.scheme(state, self.dt, self.tend)
