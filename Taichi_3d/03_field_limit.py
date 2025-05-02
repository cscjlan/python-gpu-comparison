import taichi as ti

ti.init(arch=ti.gpu)  

n = 1000

f = ti.field(dtype=ti.f32, shape=(19, n, n, n))