#!/usr/bin/env python3

from ode import OdeSolver

import numpy as np
# import scipy.fftpack as fft
import matplotlib.pyplot as plt

fft = np.fft

dt = 0.1

class Grid:
    nx, ny = (128, 128)
    xmin, xmax = (0.0, 2*np.pi)
    ymin, ymax = (0.0, 2*np.pi)
    xi = np.linspace(xmin, xmax, nx+1)
    yi = np.linspace(ymin, ymax, ny+1)
    x = (xi[1:] + xi[:-1])/2
    y = (yi[1:] + yi[:-1])/2
    dx = (xmax - xmin)/nx
    dy = (ymax - ymin)/ny
    kx1d = complex(0.0, 2*np.pi)*fft.fftfreq(nx, d=dx)
    ky1d = complex(0.0, 2*np.pi)*fft.fftfreq(ny, d=dy)

    kx = kx1d[:, np.newaxis]
    ky = ky1d[np.newaxis, :]
    ksqure = kx**2 + ky**2
    iksqure = 1/ksqure
    iksqure[0, 0] = 0.0

    @classmethod
    def ifft(cls, f):
        return np.real(fft.ifft2(f))

    @classmethod
    def fft(cls, a):
        return fft.fft2(a)

    @classmethod
    def px(cls, a, fin=True, fout=True):
        if not fin:
            a = cls.fft(a)
        a = cls.kx*a
        if not fout:
            a = cls.ifft(a)
        return a

    @classmethod
    def py(cls, a, fin=True, fout=True):
        if not fin:
            a = cls.fft(a)
        a = cls.ky*a
        if not fout:
            a = cls.ifft(a)
        return a

    @classmethod
    def laplace(cls, a, fin=True, fout=True):
        if not fin:
            a = cls.fft(a)
        a = cls.ksqure*a
        if not fout:
            a = cls.ifft(a)
        return a

    @classmethod
    def ilaplace(cls, a, fin=True, fout=True):
        if not fin:
            a = cls.fft(a)
        a = cls.iksqure*a
        a[0, 0] = 0.0
        if not fout:
            a = cls.ifft(a)
        return a

    @classmethod
    def tend_tot(cls, zeta):
        zeta = cls.fft(zeta)
        phi = cls.ilaplace(zeta)
        u = -cls.py(phi, fout=False)
        v = cls.px(phi, fout=False)
        pzpx = cls.px(zeta, fout=False)
        pzpy = cls.py(zeta, fout=False)
        print('CFL', np.max(np.sqrt(u**2 + v**2))*dt/cls.dx)
        return -u*pzpx - v*pzpy + 1.0e-4*cls.laplace(zeta, fout=False)

def read_init():
    ic = np.zeros((Grid.nx, Grid.ny))
    ymid = Grid.ny//2
    xmid = Grid.nx//2
    width = 10
    ic[:, ymid-width:ymid] = -1.0
    ic[:, ymid:ymid+width] = 1.0
    ic[xmid:xmid+1, ymid:ymid+1] += 0.1
    ic[xmid-1:xmid, ymid-1:ymid] -= 0.1
    ic.setflags(write=False)
    return ic

class Model(OdeSolver):
    ic = read_init()
    def __init__(self, dt, odescheme):
        self.grid = Grid()
        tend = self.grid.tend_tot
        super().__init__(tend, dt=dt, scheme=odescheme)

    def iter_states(self):
        return super().iter_states(self.ic)

    def plot(self, state, ax, fig):
        # state = read_init()
        # state = -Grid.py(state, fin=False, fout=False)
        # state = Grid.ilaplace(state, fin=False, fout=False)
        # state = -Grid.py(state, fin=False, fout=False)
        im = ax.contourf(self.grid.x, self.grid.y, state.T,
                cmap='seismic', levels=np.linspace(-1.0, 1.0, 32),
                extend='both')
        cb = fig.colorbar(im, ax=ax)
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cb.set_label(r'$\zeta$')
        # ax.plot(state.T)


def main():
    nstep = int(5.0e3)

    model = Model(dt=dt, odescheme='rk4')

    for i, state in zip(range(nstep), model.iter_states()):
        if i % 10 == 0:
            print(f'nstep {i:04d}')
            fig, ax = plt.subplots()
            model.plot(state, ax, fig)
            plt.savefig(f'{i:04d}.png')
            fig.clear()
            plt.close(fig)
    # plt.plot(state, label=scheme, marker='o', linestyle='')

    # plt.plot(Model.ic, label='exact', linestyle='-')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
