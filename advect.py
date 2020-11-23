#!/usr/bin/env python3

from functools import partial
from typing import AnyStr
from contextlib import contextmanager
import time

import numpy as np
import scipy.fftpack as fft
from netCDF4 import Dataset
import matplotlib.pyplot as plt

from ode import OdeSolver

dt = 0.02


@contextmanager
def Timer(tag=''):
    start = time.time()
    try:
        yield
    finally:
        tot = time.time() - start
        print(f'{tag:s} {tot:.02f}')

class Grid:
    nx = 360
    xmin = 0.0
    xmax = 360.0
    xi = np.linspace(xmin, xmax, nx+1)
    x = (xi[1:] + xi[:-1])/2
    dx = (xmax - xmin)/nx

    def __init__(self, scheme:AnyStr):
        self.scheme = getattr(self, '_scheme_'+scheme)

    def tend(self, s, u):
        return self.scheme(s, u)

    @classmethod
    def _scheme_fd(cls, s, u):
        f = s*u # flux
        weights = (
            (2, -1/12),
            (1, 2/3),
            (-1, -2/3),
            (-2, 1/12)
        )
        return sum(w*np.roll(f, shit) for shit, w in weights)/cls.dx

    @classmethod
    def _scheme_spec(cls, s, u):
        f = s*u # flux
        fspec = fft.fft(f)
        freq = fft.fftfreq(len(f), d=cls.dx)
        dfdxspec = -fspec*complex(0, 2*np.pi)*freq
        return np.real(fft.ifft(dfdxspec))

    @classmethod
    def _scheme_fv(cls, s, u):
        f = s*u # flux
        fs1 = np.roll(f, 1)
        fsn1 = np.roll(f, -1)
        c = u*dt/cls.dx
        r = (f - fs1)/(fsn1 - f + 1.0e-6)
        phi = np.maximum(0.0, np.minimum(2*r, 1.0))
        phi = np.maximum(np.minimum(r, 2.0), phi)

        fmid = f + phi*((1-c)/2)*(fsn1 - f)
        return -np.diff(fmid, prepend=fmid[-1])/cls.dx
        # plt.plot(phi, label='Zhang')
        # plt.plot(np.reshape(np.loadtxt('data.txt'), (-1, )), label='Wu')
        # plt.legend()
        # plt.show()
        # import sys; sys.exit()

def read_init():
    with Dataset('ic_homework3.nc', 'r') as dset:
        ic = dset.variables['N'][:]
    ic = np.array(ic)
    ic.setflags(write=False)
    return ic


class Model(OdeSolver):
    ic = read_init()
    def __init__(self, scheme):
        tend = partial(Grid(scheme).tend, u=10.0)
        odescheme = 'euler' if scheme == 'fv' else 'rk4'
        super().__init__(tend, dt=dt, scheme=odescheme)

    def iter_states(self):
        return super().iter_states(self.ic)

def main():
    nstep = int(1.8e4)

    plt.plot(Model.ic, label='exact', linestyle='-', linewidth=4.0)

    for scheme in ('fd', 'spec', 'fv'):
        model = Model(scheme)

        with Timer(scheme):
            for i, state in zip(range(nstep), model.iter_states()):
                if i %100 == 0:
                    print(scheme, f'nstep {i:04d}')
        plt.plot(state, label=scheme, marker='', linestyle='-', markersize=0.2)

    plt.legend()
    plt.xlabel(r'Lontitude ($^\circ$)')
    plt.ylabel('N')
    plt.title(f'NSTEP = {nstep}')
    plt.savefig(f'{nstep:d}steps.eps')

if __name__ == '__main__':
    main()
