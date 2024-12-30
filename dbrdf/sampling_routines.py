import numpy as np
import matplotlib.pyplot as plt

# We provide Importance Sampling PDF and CDFs for the following BRDF derivatives
# 1. Anisotropic GGX  -- Product Decomposition
# 1.1 Derivative with alpha_x
# 1.2 Derivative with alpha_y obtained by swapping alpha_x, alpha_y and sin_phi, cos_phi

# 2. Anisotropic Beckmann  -- Product Decomposition
# 2.1 Derivative with alpha_x
# 2.2 Derivative with alpha_y obtained by swapping alpha_x, alpha_y and sin_phi, cos_phi

# 3. Ashikhmin-Shirley  -- Product Decomposition
# 3.1 Derivative with nu
# 3.2 Derivative with nv obtained by swapping nu, nv and sin_phi, cos_phi

# 4. Microfacet ABC  -- Product Decomposition
# 4.1 Derivative with B
# 4.2 Derivative with C

# 5. Oren-Nayar  -- Mixture Decomposition
# 5.1 Derivative with Sigma

# 6. Hanrahan-Krueger (Henyey-Greenstein) -- Positivization
# 6.1 Derivative with g (positive lobe)
# 6.2 Derivative with g (negative lobe)

# 7. Burley Diffuse BSSRDF - Product Decomposition
# 7.1 Derivative with d

## GGX Theta + Phi Sampling
class AnisoGGX:
    alpha_x = -1
    alpha_y = -1
    def __init__(self, alpha_x, alpha_y):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
    
    def a_phi(self, phi):
        return (np.cos(phi)/self.alpha_x)**2 + (np.sin(phi)/self.alpha_y)**2
    
    def eval_pdf_theta_giv_phi(self, theta, phi, solid_angle=True):
        tan = np.tan(theta)
        sec = 1.0/np.cos(theta)
        a = self.a_phi(phi)
        sin_factor = 1 if solid_angle else np.sin(theta)
        return 4.0*(a**2)*(tan**2)*(sec**3)*sin_factor/((tan**2)*a + 1.0)**3
    
    def eval_cdf_theta_giv_phi(self, theta, phi):
        a = self.a_phi(phi)
        return a**2/ (a**2 - 1) - a**2*((1-a)*np.cos(4*theta) + a + 3)/(4*(a**2 - 1)*((a - 1)*np.sin(theta)**2 + 1)**2)
    
    def eval_pdf_phi(self, phi):
        return 4*np.cos(phi)**2/(np.pi*self.alpha_x**3*self.alpha_y*self.a_phi(phi)**2)
    
    def eval_cdf_phi(self, phi):
        ax = self.alpha_x; ay = self.alpha_y
        return 2/np.pi*(
            np.arctan(ax/ay * np.tan(phi)) +
            ax*ay*np.sin(2*phi)/(ax**2 + ay**2 + (ay**2 - ax**2)*np.cos(2*phi))
        )

# ## Assert that Theta PDF integrates to 1
eps = 1e-5
err_eps = 1e-2
theta_max = np.pi/2 - eps
def integrate_theta(obj, N, phi, show_plot=True):
    theta = np.linspace(0, theta_max, N)
    y = obj.eval_pdf_theta_giv_phi(theta, phi, solid_angle=False)
    if show_plot:
        plt.plot(theta, y)
        plt.show()
    int_err = np.abs(np.sum(y) * theta_max / N - 1.0)
    if int_err  > err_eps:
        print(int_err)
        print(err_eps)
        assert False


# ## Assert that Phi PDF integrates to 1
def integrate_phi(obj, N, phi_max, show_plot=True):
    phi = np.linspace(0, phi_max - eps, N)
    y = obj.eval_pdf_phi(phi)
    if show_plot:
        plt.plot(phi, y)
        plt.show()
        print(np.sum(y) * phi_max / N)
    int_err = np.abs(np.sum(y) * phi_max / N - 1.0)
    if int_err  > err_eps:
        print(int_err)
        print(err_eps)
        assert False


# ## Assert that derivative of CDF is PDF
err_eps_cdf = 5e-2
frac_fail_tol = 1e-2
def pdf_cdf_consistency_theta(obj, N, phi, show_plot=True):
    theta = np.linspace(0, theta_max, N)
    cdf = obj.eval_cdf_theta_giv_phi(theta, phi)
    theta_cdiff = theta[1:-1]
    pdf_est = (cdf[2:] - cdf[:-2])/(2*(theta_max/N))
    pdf = obj.eval_pdf_theta_giv_phi(theta_cdiff, phi, solid_angle=False)
    if show_plot:
        plt.plot(theta_cdiff, pdf, label="True PDF", alpha=0.5)
        plt.plot(theta_cdiff, pdf_est, label="PDF est", alpha=0.8)
        # plt.plot(theta_cdiff, np.abs(pdf-pdf_est)/pdf, label="err")
        plt.show()
    frac = np.sum(np.abs(pdf-pdf_est)/(pdf + 1e-5) < err_eps_cdf) / pdf.shape[0]
    # print(frac)
    assert frac > 1 - frac_fail_tol

def pdf_cdf_consistency_phi(obj, N, phi_max, show_plot=True):
    phi = np.linspace(0, phi_max, N)
    cdf = obj.eval_cdf_phi(phi)
    phi_cdiff = phi[1:-1]
    pdf_est = (cdf[2:] - cdf[:-2])/(2*(phi_max/N))
    pdf = obj.eval_pdf_phi(phi_cdiff)
    if show_plot:
        plt.plot(phi_cdiff, pdf_est, label="PDF est")
        plt.plot(phi_cdiff, pdf, label="True PDF")
        plt.show()
    frac = np.sum(np.abs(pdf-pdf_est)/pdf < err_eps_cdf) / pdf.shape[0]
    # print(frac)
    assert frac > 1 - frac_fail_tol

num_intervals = 100000
num_runs = 100

for _ in range(num_runs):
    alpha_x = np.clip(np.random.rand()**2, 0.005, 1)
    alpha_y = np.clip(np.random.rand()**2, 0.005, 1)
    phi = np.random.rand()*2*np.pi
    obj = AnisoGGX(alpha_x, alpha_y)
    val = integrate_theta(obj, num_intervals, phi, show_plot=False)

for _ in range(num_runs):
    alpha_x = np.clip(np.random.rand()**2, 0.005, 1)
    alpha_y = np.clip(np.random.rand()**2, 0.005, 1)
    phi = np.random.rand()*2*np.pi
    obj = AnisoGGX(alpha_x, alpha_y)
    val = pdf_cdf_consistency_theta(obj, num_intervals, phi, show_plot=False)    

for idx in range(num_runs):
    alpha_x = np.clip(np.random.rand()**2, 0.005, 1)
    alpha_y = np.clip(np.random.rand()**2, 0.005, 1)
    obj = AnisoGGX(alpha_x, alpha_y)
    # print(alpha_x, alpha_y)
    val = integrate_phi(obj, num_intervals, phi_max=np.pi/2, show_plot=False)


for _ in range(num_runs):
    alpha_x = np.clip(np.random.rand()**2, 0.005, 1)
    alpha_y = np.clip(np.random.rand()**2, 0.005, 1)
    obj = AnisoGGX(alpha_x, alpha_y)
    val = pdf_cdf_consistency_phi(obj, num_intervals, phi_max=np.pi/2, show_plot=False)    


## Beckmann Theta Sampling
class AnisoBeckmann:
    alpha_x = -1
    alpha_y = -1
    def __init__(self, alpha_x, alpha_y):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
    
    def a_phi(self, phi):
        return (np.cos(phi)/self.alpha_x)**2 + (np.sin(phi)/self.alpha_y)**2
    
    def eval_pdf_theta_giv_phi(self, theta, phi, solid_angle=True):
        tan = np.tan(theta)
        sec = 1.0/np.cos(theta)
        a = self.a_phi(phi)
        sin_factor = 1 if solid_angle else np.sin(theta)
        return 2.0*(a**2)*(tan**2)*(sec**3)*sin_factor*np.exp(-a*tan**2)
    
    def eval_cdf_theta_giv_phi(self, theta, phi):
        a = self.a_phi(phi)
        tan = np.tan(theta)
        return 1 - (1 + a*tan**2)*np.exp(-a*tan**2)
    # Phi routines same as GGX


for _ in range(num_runs):
    alpha_x = np.clip(np.random.rand()**2, 0.01, 1)
    alpha_y = np.clip(np.random.rand()**2, 0.01, 1)
    phi = np.random.rand()*2*np.pi
    obj = AnisoBeckmann(alpha_x, alpha_y)
    val = integrate_theta(obj, num_intervals, phi, show_plot=False)


for _ in range(num_runs):
    alpha_x = np.clip(np.random.rand()**2, 0.01, 1)
    alpha_y = np.clip(np.random.rand()**2, 0.01, 1)
    phi = np.random.rand()*2*np.pi
    obj = AnisoBeckmann(alpha_x, alpha_y)
    val = pdf_cdf_consistency_theta(obj, num_intervals, phi, show_plot=False)    


## AshikhminShirley Theta + Phi Sampling
class AshikhminShirley:
    nu = -1
    nv = -1
    def __init__(self, nu, nv):
        self.nu = nu
        self.nv = nv
    
    def a_phi(self, phi):
        return self.nu*np.cos(phi)**2 + self.nv*np.sin(phi)**2
    
    def eval_pdf_phi(self, phi):
        a = self.a_phi(phi)
        return 4.0*(self.nu+1)**1.5*(self.nv+1)**0.5*np.cos(phi)**2/(np.pi*(1 + a)**2)
    
    def eval_cdf_phi(self, phi):
        nu = self.nu; nv = self.nv
        return 2/np.pi*(
            np.arctan(np.sqrt((nv+1)/(nu+1)) * np.tan(phi)) +
            np.sqrt((nu+1)*(nv+1))*np.sin(2*phi)/(nu + nv + 2 + (nu - nv)*np.cos(2*phi))
        )
    
    def eval_pdf_theta_giv_phi(self, theta, phi, solid_angle=True):
        cos = np.cos(theta)
        a = self.a_phi(phi)
        sin_factor = 1 if solid_angle else np.sin(theta)
        return -np.log(cos)*(1+a)**2*cos**a*sin_factor
    
    def eval_cdf_theta_giv_phi(self, theta, phi):
        a = self.a_phi(phi)
        cos = np.cos(theta)
        return 1 - (1 - (a+1)*np.log(cos))*cos**(a+1)


for idx in range(num_runs):
    nu = np.clip(np.random.rand()*50, 1, 50)
    nv = np.clip(np.random.rand()*50, 1, 50)
    obj = AshikhminShirley(nu, nv)
    val = integrate_phi(obj, num_intervals, phi_max=np.pi/2, show_plot=False)

for _ in range(num_runs):
    nu = np.clip(np.random.rand()*50, 1, 50)
    nv = np.clip(np.random.rand()*50, 1, 50)
    obj = AshikhminShirley(nu, nv)
    val = pdf_cdf_consistency_phi(obj, num_intervals, phi_max=np.pi/2, show_plot=False)        


for _ in range(num_runs):
    nu = np.clip(np.random.rand()*50, 1, 50)
    nv = np.clip(np.random.rand()*50, 1, 50)
    obj = AshikhminShirley(nu, nv)
    phi = np.random.rand()*2*np.pi
    val = integrate_theta(obj, num_intervals, phi, show_plot=False)


for _ in range(num_runs):
    nu = np.clip(np.random.rand()*50, 1, 50)
    nv = np.clip(np.random.rand()*50, 1, 50)
    obj = AshikhminShirley(nu, nv)
    phi = np.random.rand()*2*np.pi
    val = pdf_cdf_consistency_theta(obj, num_intervals, phi, show_plot=False)    

## Microfacet ABC Theta Sampling for derivative w B
class ABC_B:
    def __init__(self, b, c):
        self.b = b
        self.c = c
    # independent of phi
    def eval_pdf_theta_giv_phi(self, theta, phi, solid_angle=True):
        sin_factor = 1 if solid_angle else np.sin(theta)
        b = self.b; c = self.c;
        cos = np.cos(theta)
        return b**2*c*(c-1)*(b+1)**c*(cos - 1)*(1 + b*(1 - cos))**(-1-c)*sin_factor/(1 + b*c - (b+1)**c)
    
    # independent of phi
    def eval_cdf_theta_giv_phi(self, theta, phi):
        b = self.b; c = self.c;
        cos = np.cos(theta)
        return ((b+1)**c*(1+b*(1-cos))**(-c)*(1+b*c*(1-cos)) - (1+b)**c)/(1 + b*c - (1 + b)**c)

class ABC_C:
    def __init__(self, b, c):
        self.b = b
        self.c = c
    # independent of phi
    def eval_pdf_theta_giv_phi(self, theta, phi, solid_angle=True):
        sin_factor = 1 if solid_angle else np.sin(theta)
        b = self.b; c = self.c;
        cos = np.cos(theta)
        return b*(c-1)**2 * np.log(1 + b*(1-cos))*sin_factor / (1 - (1+b)**(1-c)*((c-1)*np.log(1+b)+1)) / (1+ b*(1 -cos))**c
    
    # independent of phi
    def eval_cdf_theta_giv_phi(self, theta, phi):
        b = self.b; c = self.c;
        cos = np.cos(theta)
        return (1 - (1+b*(1-cos))**(1-c)*((c-1)*np.log(1 + b*(1 - cos)) + 1))/(1 - (1+b)**(1-c)*((c-1)*np.log(1+b) + 1))
    


for _ in range(num_runs):
    b = np.clip(np.random.rand()*1000, 1, 1000)
    c = np.clip(np.random.rand()*2, 0.1, 2)
    obj = ABC_B(b, c)
    phi = np.random.rand()*2*np.pi
    val = integrate_theta(obj, num_intervals, phi, show_plot=False)

for _ in range(num_runs):
    b = np.clip(np.random.rand()*1000, 1, 1000)
    c = np.clip(np.random.rand()*2, 0.1, 2)
    obj = ABC_B(b, c)
    phi = np.random.rand()*2*np.pi
    val = pdf_cdf_consistency_theta(obj, num_intervals, phi, show_plot=False)        


for _ in range(num_runs):
    b = np.clip(np.random.rand()*1000, 1, 1000)
    c = np.clip(np.random.rand()*2, 0.1, 2)
    obj = ABC_C(b, c)
    phi = np.random.rand()*2*np.pi
    val = integrate_theta(obj, num_intervals, phi, show_plot=False)


for _ in range(num_runs):
    b = np.clip(np.random.rand()*1000, 1, 1000)
    c = np.clip(np.random.rand()*2, 0.1, 2)
    obj = ABC_C(b, c)
    phi = np.random.rand()*2*np.pi
    val = pdf_cdf_consistency_theta(obj, num_intervals, phi, show_plot=False)        


## Oren Nayar Second Term, first term is just Cosine Hemispherical Sampling
class OrenNayar:
    def __init__(self, sigma):
        self.sigma = sigma
    
    # independent of phi
    def eval_pdf_theta_giv_phi(self, th_i, th_o, solid_angle=True):
        sin_factor = 1 if solid_angle else np.sin(th_i)
        A21 = 0.5*np.sin(th_o)*(th_o - np.sin(th_o)*np.cos(th_o))
        A22 = 1.0/3.0*np.tan(th_o)*(1 - np.sin(th_o)**3)
        T2 = A21 + A22
        A21p = A21/T2; A22p = A22/T2;
        pdf1 = A21p*np.sin(th_i)/(0.5*(th_o - np.sin(th_o)*np.cos(th_o)))
        pdf2 = A22p*3.0*np.sin(th_i)*np.cos(th_i)/(1 - np.sin(th_o)**3)
        return np.where(th_i < th_o, pdf1, pdf2)*sin_factor
    
    # independent of phi
    def eval_cdf_theta_giv_phi(self, th_i, th_o):
        A21 = 0.5*np.sin(th_o)*(th_o - np.sin(th_o)*np.cos(th_o))
        A22 = 1.0/3.0*np.tan(th_o)*(1 - np.sin(th_o)**3)
        T2 = A21 + A22
        A21p = A21/T2; A22p = A22/T2;
        cdf1 = (th_i - np.sin(th_i)*np.cos(th_i))/(th_o - np.sin(th_o)*np.cos(th_o))
        cdf2 = (np.sin(th_i)**3 - np.sin(th_o)**3)/(1 - np.sin(th_o)**3)
        return np.where(th_i < th_o, A21p*cdf1, A21p + A22p*cdf2)

for _ in range(num_runs):
    sigma = np.clip(np.random.rand(), 1e-2, 1-1e-2)
    obj = OrenNayar(sigma)
    th_o = np.random.rand()*np.pi/2
    val = integrate_theta(obj, num_intervals, th_o, show_plot=False)

for _ in range(num_runs):
    sigma = np.clip(np.random.rand(), 1e-2, 1-1e-2)
    obj = OrenNayar(sigma)
    th_o = np.random.rand()*np.pi/2
    val = pdf_cdf_consistency_theta(obj, num_intervals, th_o, show_plot=False)        



class HenyeyGreensteinPlus:
    def __init__(self, g):
        self.g = g
        self.C = 3**(3/2)*(1-g**2)/((3+g**2)**(3/2) - 3**(3/2)*(1-g**2))
    
    # independent of phi
    def eval_pdf_theta_giv_phi(self, th_i, th_o, solid_angle=True):
        sin_factor = 1 if solid_angle else np.sin(th_i)
        cos = np.cos(th_i)
        return np.maximum(self.C*g**2 * ((g**2 + 3)*cos + g*(g**2 - 5))/(g**2 - 2*g*cos + 1)**(5/2), 0)*sin_factor
    
    def eval_cdf_theta_giv_phi(self, th_i, th_o):
        g = self.g
        cos = np.cos(th_i)
        pdf = self.eval_pdf_theta_giv_phi(th_i, th_o, False)
        return np.where(pdf > 0, self.C*(3*g**2+1 - g*(3+g**2)*cos)/(g**2-2*g*cos+1)**(3/2) - 1, 1)
    

class HenyeyGreensteinMinus:
    def __init__(self, g):
        self.g = g
        self.C = 3**(3/2)*(1-g**2)/((3+g**2)**(3/2) - 3**(3/2)*(1-g**2))
    
    # independent of phi
    def eval_pdf_theta_giv_phi(self, th_i, th_o, solid_angle=True):
        sin_factor = 1 if solid_angle else np.sin(th_i)
        cos = np.cos(th_i)
        return -np.minimum(self.C*g**2 * ((g**2 + 3)*cos + g*(g**2 - 5))/(g**2 - 2*g*cos + 1)**(5/2), 0)*sin_factor
    
    def eval_cdf_theta_giv_phi(self, th_i, th_o):
        g = self.g
        cos = np.cos(th_i)
        pdf = self.eval_pdf_theta_giv_phi(th_i, th_o, False)
        return np.where(pdf > 0, (1-self.C)-self.C*(3*g**2+1 - g*(3+g**2)*cos)/(g**2-2*g*cos+1)**(3/2) - 1, 0)

def integrate_theta_sphere(obj, N, phi, show_plot=True):
    theta = np.linspace(0, np.pi - eps, N)
    y = obj.eval_pdf_theta_giv_phi(theta, phi, solid_angle=False)
    if show_plot:
        plt.plot(theta, y)
        plt.show()
    int_err = np.abs(np.sum(y) * np.pi / N - 1.0)
    if int_err  > err_eps:
        print(int_err)
        print(err_eps)
        assert False

def pdf_cdf_consistency_theta_sphere(obj, N, phi, show_plot=True):
    theta = np.linspace(0, np.pi, N)
    cdf = obj.eval_cdf_theta_giv_phi(theta, phi)
    theta_cdiff = theta[1:-1]
    pdf_est = (cdf[2:] - cdf[:-2])/(2*(np.pi/N))
    pdf = obj.eval_pdf_theta_giv_phi(theta_cdiff, phi, solid_angle=False)
    if show_plot:
        plt.plot(theta_cdiff, pdf, label="True PDF", alpha=0.5)
        plt.plot(theta_cdiff, pdf_est, label="PDF est", alpha=0.8)
        plt.legend()
        plt.show()
    frac = np.sum(np.abs(pdf-pdf_est)/(pdf + 1e-5) < err_eps_cdf) / pdf.shape[0]
    assert frac > 1 - frac_fail_tol


for _ in range(num_runs):
    g = np.clip(np.random.rand()*2 - 1, -1+1e-3, 1-1e-3)
    obj = HenyeyGreensteinPlus(g)
    th_o = np.random.rand()*np.pi/2
    val = integrate_theta_sphere(obj, num_intervals, th_o, show_plot=False)


for _ in range(num_runs):
    g = np.clip(np.random.rand()*2 - 1, -1+1e-3, 1-1e-3)
    obj = HenyeyGreensteinPlus(g)
    th_o = np.random.rand()*np.pi/2
    val = pdf_cdf_consistency_theta_sphere(obj, num_intervals, th_o, show_plot=False)        

for _ in range(num_runs):
    g = np.clip(np.random.rand()*2 - 1, -1+1e-3, 1-1e-3)
    obj = HenyeyGreensteinMinus(g)
    th_o = np.random.rand()*np.pi/2
    val = integrate_theta_sphere(obj, num_intervals, th_o, show_plot=False)

for _ in range(num_runs):
    g = np.clip(np.random.rand()*2 - 1, -1+1e-3, 1-1e-3)
    obj = HenyeyGreensteinMinus(g)
    th_o = np.random.rand()*np.pi/2
    val = pdf_cdf_consistency_theta_sphere(obj, num_intervals, th_o, show_plot=False)        


class BurleyDiffuseBSSRDF:
    def __init__(self, d):
        self.d = d
    
    def eval_pdf(self, r, multiply_r=True):
        jacobian = 1 if not multiply_r else r
        return (np.exp(-r/d) + np.exp(-r/(3*d))/3)/(4*d**2) * jacobian
    
    def eval_cdf(self, r):
        return 1 - np.exp(-r/d)*(r+d)/(4*d) - np.exp(-r/(3*d))*(3*d + r)/(4*d)
    



def integrate_r(obj, N, show_plot=True):
    lim = 10**5
    r = np.linspace(0, lim, N)
    y = obj.eval_pdf(r, multiply_r=True)
    if show_plot:
        plt.plot(r, y)
        plt.show()
    int_err = np.abs(np.sum(y) * lim / N - 1.0)
    if int_err  > err_eps:
        print(int_err)
        print(err_eps)
        assert False


def pdf_cdf_consistency_r(obj, N, show_plot=True):
    lim = 10**5
    r = np.linspace(0, lim, N)
    cdf = obj.eval_cdf(r)
    r_cdiff = r[1:-1]
    pdf_est = (cdf[2:] - cdf[:-2])/(2*(lim/N))
    pdf = obj.eval_pdf(r_cdiff, multiply_r=True)
    if show_plot:
        plt.plot(r_cdiff, pdf, label="True PDF", alpha=0.5)
        plt.plot(r_cdiff, pdf_est, label="PDF est", alpha=0.8)
        plt.legend()
        plt.show()
    frac = np.sum(np.abs(pdf-pdf_est)/(pdf + 1e-5) < err_eps_cdf) / pdf.shape[0]
    assert frac > 1 - frac_fail_tol


for _ in range(num_runs):
    d = np.clip(np.random.rand()*10**3, 0.1, 1000)
    obj = BurleyDiffuseBSSRDF(d)
    val = integrate_r(obj, num_intervals, show_plot=False)


for _ in range(num_runs):
    d = np.clip(np.random.rand()*10**3, 0.1, 1000)
    obj = BurleyDiffuseBSSRDF(d)
    val = pdf_cdf_consistency_r(obj, num_intervals, show_plot=False)

