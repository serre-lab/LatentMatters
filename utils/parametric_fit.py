from numpy import *
from numpy.polynomial import polynomial as pl
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from scipy import optimize

## helper function
def uniform_param(P):
    u = linspace(0, 1, len(P))
    return u


def chordlength_param(P):
    u = generate_param(P, alpha=1.0)
    return u


def centripetal_param(P):
    u = generate_param(P, alpha=0.5)
    return u


def generate_param(P, alpha):
    n = len(P)
    u = zeros(n)
    u_sum = 0
    for i in range(1, n):
        u_sum += linalg.norm(P[i, :] - P[i - 1, :]) ** alpha
        u[i] = u_sum

    return u / max(u)


def find_min_gss(f, a, b, eps=1e-4):
    # Golden section: 1/phi = 2/(1+sqrt(5))
    R = 0.61803399

    # Num of needed iterations to get precision eps: log(eps/|b-a|)/log(R)
    n_iter = int(ceil(-2.0780869 * log(eps / abs(b - a))))
    c = b - (b - a) * R
    d = a + (b - a) * R

    for i in range(n_iter):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) * R
        d = a + (b - a) * R

    return (b + a) / 2



def iterative_param(P, u, fxcoeff, fycoeff, fzcoeff, fig_ax, iter_i, plt_color, plt_alpha):

    u_new = u.copy()
    f_u = zeros(3)

    # --- Calculate approx. error s(u) related to point P_i
    def calc_s(u):
        f_u[0] = pl.polyval(u, fxcoeff)
        f_u[1] = pl.polyval(u, fycoeff)
        f_u[2] = pl.polyval(u, fzcoeff)

        s_u = linalg.norm(P[i] - f_u)
        return s_u

    # --- Find new values u that locally minimising the approximation error (excl. fixed end-points)
    for i in range(1, len(u) - 1):
        # --- Find new u_i minimising s(u_i) by Golden search method
        u_new[i] = find_min_gss(calc_s, u[i - 1], u[i + 1])

        # --- Sample some values bewteen u[i-1] and u[i+1] to plot graph
        u_samp = linspace(u[i - 1], u[i + 1], 25)

        x = pl.polyval(u_samp, fxcoeff)
        y = pl.polyval(u_samp, fycoeff)
        z = pl.polyval(u_samp, fzcoeff)

        residual = P[i] - array([x, y, z]).T
        s_u_samp = [linalg.norm(residual[j]) for j in range(len(u_samp))]

        # --- Plot error to given axes
        if fig_ax is not None:
            fig_ax.plot(u_samp, s_u_samp, color=plt_color[iter_i], alpha=plt_alpha)
            fig_ax.plot(u_new[i], calc_s(u_new[i]), 'o', color=plt_color[iter_i], alpha=plt_alpha)

    return u_new

def iterative_param_pw(P, u, fxcoeff, fycoeff, fzcoeff, fig_ax, iter_i, plt_color, plt_alpha):

    u_new = u.copy()
    f_u = zeros(3)

    # --- Calculate approx. error s(u) related to point P_i
    def calc_s(u):
        f_u[0] = piecewise_linear(u, *fxcoeff)
        f_u[1] = piecewise_linear(u, *fycoeff)
        f_u[2] = piecewise_linear(u, *fzcoeff)

        s_u = linalg.norm(P[i] - f_u)
        return s_u

    # --- Find new values u that locally minimising the approximation error (excl. fixed end-points)
    for i in range(1, len(u) - 1):
        # --- Find new u_i minimising s(u_i) by Golden search method
        u_new[i] = find_min_gss(calc_s, u[i - 1], u[i + 1])

        # --- Sample some values bewteen u[i-1] and u[i+1] to plot graph
        u_samp = linspace(u[i - 1], u[i + 1], 25)

        x = piecewise_linear(u_samp, *fxcoeff)
        y = piecewise_linear(u_samp, *fycoeff)
        z = piecewise_linear(u_samp, *fzcoeff)

        residual = P[i] - array([x, y, z]).T
        s_u_samp = [linalg.norm(residual[j]) for j in range(len(u_samp))]

        # --- Plot error to given axes
        if fig_ax is not None:
            fig_ax.plot(u_samp, s_u_samp, color=plt_color[iter_i], alpha=plt_alpha)
            fig_ax.plot(u_new[i], calc_s(u_new[i]), 'o', color=plt_color[iter_i], alpha=plt_alpha)

    return u_new


## fit
def parametric_fit(P, analyze=False, polydeg=3, max_iter=20, eps=1e-5, label=['param', 'diversity', 'accuracy'], weight_extremum=None, title=None, down_weight=None):
    n = len(P)
    w = ones(n)  # Set weights for knot points
    if weight_extremum is not None:
        for idx in weight_extremum:
            w[idx] = 100000

    if down_weight is not None:
        for idx in down_weight:
            w[idx] = 0

    plt_alpha = 0.6
    plt_color = cm.rainbow(linspace(1, 0, max_iter))

    # -------------------------------------------------------------------------------
    # Init variables
    # -------------------------------------------------------------------------------
    f_u = zeros([n, 3])
    uu = linspace(0, 1, 100)
    f_uu = zeros([len(uu), 3])
    S_hist = []

    if analyze:
        fig = figure(figsize=(15, 15))
        figshape = (3, 3)
        ax = [None] * 5
        ax[0] = subplot2grid(figshape, loc=(0, 0))
        ax[1] = subplot2grid(figshape, loc=(0, 1))
        ax[2] = subplot2grid(figshape, loc=(0, 2))
        ax[3] = subplot2grid(figshape, loc=(1, 0), colspan=3)
        ax[4] = subplot2grid(figshape, loc=(2, 0), colspan=2)
        i = 0
        # ax[i].plot(P_gen[:,0], P_gen[:,1], 'y-', lw=2 ,label='Generating Curve')
        ax[i].plot(P[:, 0], P[:, 1], 'ks', label='Knot points P')
        ax[i].set_title('View {0} - {1}'.format(label[0], label[1]))
        ax[i].set_xlabel(label[0])
        ax[i].set_ylabel(label[1])
        # ax[i].set_aspect('equal', 'datalim')
        ax[i].margins(.1, .1)
        ax[i].grid()
        i = 1
        # ax[i].plot(P_gen[:,0], P_gen[:,2], 'y-', lw=2 ,label='Generating Curve')
        ax[i].plot(P[:, 0], P[:, 2], 'ks', label='Knot points P')
        ax[i].set_title('View {0}-{1}'.format(label[0], label[2]))
        ax[i].set_xlabel(label[0])
        ax[i].set_ylabel(label[2])
        # ax[i].set_aspect('equal', 'datalim')
        ax[i].margins(.1, .1)
        ax[i].grid()
        i = 2
        # ax[i].plot(P_gen[:,1], P_gen[:,2], 'y-', lw=2 ,label='Generating Curve')
        ax[i].plot(P[:, 1], P[:, 2], 'ks', label='Knot points P')
        ax[i].set_title('View {0}-{1}'.format(label[1], label[2]))
        ax[i].set_xlabel(label[1]);
        ax[i].set_ylabel(label[2]);
        # ax[i].set_aspect('equal', 'datalim')
        ax[i].margins(.1, .1)
        ax[i].legend()
        ax[i].grid()
        i = 3
        ax[i].set_title('Local minimization of approximation error $s(u_i)$ for each $u_i$')
        ax[i].set_xlabel('$u_i$');
        ax[i].set_ylabel('$s(u_i)$');
        ax[i].grid()
        i = 4
        ax[i].set_title('Approximation error $S$ for each iteration')
        ax[i].set_xlabel('Iteration');
        ax[i].set_ylabel('$S$');
        ax[i].grid()

    t = 1

    for iter_i in range(max_iter):

        # --- Initial or iterative parametrization
        if iter_i == 0:
            # u = uniform_param(P)
            # u = chordlength_param(P)
            u = centripetal_param(P)
        else:
            if analyze:
                u = iterative_param(P, u, fxcoeff, fycoeff, fzcoeff, ax[3], iter_i, plt_color, plt_alpha)
            else:
                u = iterative_param(P, u, fxcoeff, fycoeff, fzcoeff, None, iter_i, plt_color, plt_alpha)

        # --- Compute polynomial approximations and get their coefficients
        fxcoeff = pl.polyfit(u, P[:, 0], polydeg, w=w)
        fycoeff = pl.polyfit(u, P[:, 1], polydeg, w=w)
        fzcoeff = pl.polyfit(u, P[:, 2], polydeg, w=w)

        # --- Calculate function values f(u)=(fx(u),fy(u),fz(u))
        f_u[:, 0] = pl.polyval(u, fxcoeff)
        f_u[:, 1] = pl.polyval(u, fycoeff)
        f_u[:, 2] = pl.polyval(u, fzcoeff)

        # --- Calculate fine values for ploting
        f_uu[:, 0] = pl.polyval(uu, fxcoeff)
        f_uu[:, 1] = pl.polyval(uu, fycoeff)
        f_uu[:, 2] = pl.polyval(uu, fzcoeff)

        if analyze:
            # --- Print plots
            hp = ax[0].plot(f_uu[:, 0], f_uu[:, 1], color=plt_color[iter_i], alpha=plt_alpha)
            hp = ax[1].plot(f_uu[:, 0], f_uu[:, 2], color=plt_color[iter_i], alpha=plt_alpha)
            hp = ax[2].plot(f_uu[:, 1], f_uu[:, 2], color=plt_color[iter_i], alpha=plt_alpha)

            # --- Errors of init parametrization
            if iter_i == 0:
                for i in range(1, len(u) - 1):
                    ax[3].plot(u[i], linalg.norm(P[i] - f_u[i]), 'o', color=plt_color[iter_i], alpha=plt_alpha)

        # --- Total error of approximation S for iteration i
        S = 0
        for j in range(len(u)):
            S += w[j] * linalg.norm(P[j] - f_u[j])

        if analyze:
            # --- Add bar of approx. error
            ax[4].bar(iter_i, S, width=0.6, color=plt_color[iter_i], alpha=plt_alpha)

        S_hist.append(S)

        # --- Stop iterating if change in error is lower than desired condition
        if iter_i > 0:
            S_change = S_hist[iter_i - 1] / S_hist[iter_i] - 1
            # print('iteration:%3i, approx.error: %.4f (%f)' % (iter_i, S_hist[iter_i], S_change))
            if S_change < eps:
                break

    if analyze:
        plt.suptitle(title)
        show()

    return fxcoeff, fycoeff, fzcoeff

#def piecewise_linear(x, x0, y0, k1, k2):
#    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
def piecewise_linear(x, a, b):
    return a*x + b

def parametric_fit_pw(P, analyze=False, max_iter=20, eps=1e-5, label=['param', 'diversity', 'accuracy'], weight_extremum=False, title=None):
    n = len(P)
    w = ones(n)  # Set weights for knot points
    if weight_extremum is not None:
        for idx in weight_extremum:
            w[idx] = 100000

    plt_alpha = 0.6
    plt_color = cm.rainbow(linspace(1, 0, max_iter))

    # -------------------------------------------------------------------------------
    # Init variables
    # -------------------------------------------------------------------------------
    f_u = zeros([n, 3])
    uu = linspace(0, 1, 100)
    f_uu = zeros([len(uu), 3])
    S_hist = []

    if analyze:
        fig = figure(figsize=(15, 15))
        figshape = (3, 3)
        ax = [None] * 5
        ax[0] = subplot2grid(figshape, loc=(0, 0))
        ax[1] = subplot2grid(figshape, loc=(0, 1))
        ax[2] = subplot2grid(figshape, loc=(0, 2))
        ax[3] = subplot2grid(figshape, loc=(1, 0), colspan=3)
        ax[4] = subplot2grid(figshape, loc=(2, 0), colspan=2)
        i = 0
        # ax[i].plot(P_gen[:,0], P_gen[:,1], 'y-', lw=2 ,label='Generating Curve')
        ax[i].plot(P[:, 0], P[:, 1], 'ks', label='Knot points P')
        ax[i].set_title('View {0} - {1}'.format(label[0], label[1]))
        ax[i].set_xlabel(label[0])
        ax[i].set_ylabel(label[1])
        # ax[i].set_aspect('equal', 'datalim')
        ax[i].margins(.1, .1)
        ax[i].grid()
        i = 1
        # ax[i].plot(P_gen[:,0], P_gen[:,2], 'y-', lw=2 ,label='Generating Curve')
        ax[i].plot(P[:, 0], P[:, 2], 'ks', label='Knot points P')
        ax[i].set_title('View {0}-{1}'.format(label[0], label[2]))
        ax[i].set_xlabel(label[0])
        ax[i].set_ylabel(label[2])
        # ax[i].set_aspect('equal', 'datalim')
        ax[i].margins(.1, .1)
        ax[i].grid()
        i = 2
        # ax[i].plot(P_gen[:,1], P_gen[:,2], 'y-', lw=2 ,label='Generating Curve')
        ax[i].plot(P[:, 1], P[:, 2], 'ks', label='Knot points P')
        ax[i].set_title('View {0}-{1}'.format(label[1], label[2]))
        ax[i].set_xlabel(label[1]);
        ax[i].set_ylabel(label[2]);
        # ax[i].set_aspect('equal', 'datalim')
        ax[i].margins(.1, .1)
        ax[i].legend()
        ax[i].grid()
        i = 3
        ax[i].set_title('Local minimization of approximation error $s(u_i)$ for each $u_i$')
        ax[i].set_xlabel('$u_i$');
        ax[i].set_ylabel('$s(u_i)$');
        ax[i].grid()
        i = 4
        ax[i].set_title('Approximation error $S$ for each iteration')
        ax[i].set_xlabel('Iteration');
        ax[i].set_ylabel('$S$');
        ax[i].grid()

    t = 1

    for iter_i in range(max_iter):

        # --- Initial or iterative parametrization
        if iter_i == 0:
            # u = uniform_param(P)
            # u = chordlength_param(P)
            u = centripetal_param(P)
        else:
            if analyze:
                u = iterative_param_pw(P, u, fxcoeff, fycoeff, fzcoeff, ax[3], iter_i, plt_color, plt_alpha)
            else:
                u = iterative_param_pw(P, u, fxcoeff, fycoeff, fzcoeff, None, iter_i, plt_color, plt_alpha)

        # --- Compute polynomial approximations and get their coefficients
        #fxcoeff = pl.polyfit(u, P[:, 0], polydeg, w=w)
        fxcoeff, e = optimize.curve_fit(piecewise_linear, u,  P[:, 0])
        fycoeff, e = optimize.curve_fit(piecewise_linear, u, P[:, 1])
        fzcoeff, e = optimize.curve_fit(piecewise_linear, u, P[:, 2])
        #fycoeff = pl.polyfit(u, P[:, 1], polydeg, w=w)
        #fzcoeff = pl.polyfit(u, P[:, 2], polydeg, w=w)

        # --- Calculate function values f(u)=(fx(u),fy(u),fz(u))
        #f_u[:, 0] = pl.polyval(u, fxcoeff)
        #f_u[:, 1] = pl.polyval(u, fycoeff)
        #f_u[:, 2] = pl.polyval(u, fzcoeff)

        f_u[:, 0] = piecewise_linear(u, *fxcoeff)
        f_u[:, 1] = piecewise_linear(u, *fycoeff)
        f_u[:, 2] = piecewise_linear(u, *fzcoeff)


        # --- Calculate fine values for ploting
        #f_uu[:, 0] = pl.polyval(uu, fxcoeff)
        #f_uu[:, 1] = pl.polyval(uu, fycoeff)
        #f_uu[:, 2] = pl.polyval(uu, fzcoeff)

        f_uu[:, 0] = piecewise_linear(uu, *fxcoeff)
        f_uu[:, 1] = piecewise_linear(uu, *fycoeff)
        f_uu[:, 2] = piecewise_linear(uu, *fzcoeff)

        if analyze:
            # --- Print plots
            hp = ax[0].plot(f_uu[:, 0], f_uu[:, 1], color=plt_color[iter_i], alpha=plt_alpha)
            hp = ax[1].plot(f_uu[:, 0], f_uu[:, 2], color=plt_color[iter_i], alpha=plt_alpha)
            hp = ax[2].plot(f_uu[:, 1], f_uu[:, 2], color=plt_color[iter_i], alpha=plt_alpha)

            # --- Errors of init parametrization
            if iter_i == 0:
                for i in range(1, len(u) - 1):
                    ax[3].plot(u[i], linalg.norm(P[i] - f_u[i]), 'o', color=plt_color[iter_i], alpha=plt_alpha)

        # --- Total error of approximation S for iteration i
        S = 0
        for j in range(len(u)):
            S += w[j] * linalg.norm(P[j] - f_u[j])

        if analyze:
            # --- Add bar of approx. error
            ax[4].bar(iter_i, S, width=0.6, color=plt_color[iter_i], alpha=plt_alpha)

        S_hist.append(S)

        # --- Stop iterating if change in error is lower than desired condition
        if iter_i > 0:
            S_change = S_hist[iter_i - 1] / S_hist[iter_i] - 1
            # print('iteration:%3i, approx.error: %.4f (%f)' % (iter_i, S_hist[iter_i], S_change))
            if S_change < eps:
                break

    if analyze:
        plt.suptitle(title)
        show()

    return fxcoeff, fycoeff, fzcoeff

