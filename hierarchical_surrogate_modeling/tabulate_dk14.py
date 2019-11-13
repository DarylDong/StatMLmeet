"""
"""
import numpy as np
import argparse
from colossus.cosmology import cosmology
from colossus.halo import profile_dk14
from pyDOE2 import lhs
from time import time


def _get_formatted_output_line(outline):
    """
    """
    _output = np.array_str(outline, suppress_small=True, precision=5)
    output = ' '.join(_output.replace('[', '').replace(']', '').split()) + '\n'
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("ntrain",
        help="Number of parameter space points to tabulate", type=int)
    parser.add_argument("-outname",
        help="Filename for the tabulation", default='DATA/dk14_ds_tabulation')
    parser.add_argument("-npts",
        help="Number of points tabulated for each profile", type=int, default=200)
    parser.add_argument("-cosmo",
        help="Assumed cosmology", default='planck15')
    parser.add_argument("-rhos",
        help="Constant value for rhos", type=int, default=1e10)

    parser.add_argument("-rt", help="Constant value of rt", default=1000.)

    parser.add_argument("-rs_by_rt_lo", help="Lower bound on rs/rt", default=0.01)
    parser.add_argument("-rs_by_rt_hi", help="Upper bound on rs/rt", default=0.75)

    parser.add_argument("-alpha_lo", help="Lower bound on alpha", default=0.05)
    parser.add_argument("-alpha_hi", help="Upper bound on alpha", default=1)

    parser.add_argument("-beta_lo", help="Lower bound on beta", default=1)
    parser.add_argument("-beta_hi", help="Upper bound on beta", default=10)

    parser.add_argument("-gamma_lo", help="Lower bound on gamma", default=1)
    parser.add_argument("-gamma_hi", help="Upper bound on gamma", default=10)

    args = parser.parse_args()
    outname = args.outname
    ntrain, npts = args.ntrain, args.npts
    rhos, rt = args.rhos, args.rt
    cosmo = args.cosmo
    rs_lo, rs_hi = args.rs_by_rt_lo, args.rs_by_rt_hi
    alpha_lo, alpha_hi = args.alpha_lo, args.alpha_hi
    beta_lo, beta_hi = args.beta_lo, args.beta_hi
    gamma_lo, gamma_hi = args.gamma_lo, args.gamma_hi

    r_by_rt = np.logspace(-2, np.log10(2), npts)
    ndim = 4
    design = lhs(ndim, ntrain, criterion='maximin')

    rs_arr = rt*design[:, 0]*(rs_hi-rs_lo)+rs_lo
    alpha_arr = design[:, 1]*(alpha_hi-alpha_lo)+alpha_lo
    beta_arr = design[:, 2]*(beta_hi-beta_lo)+beta_lo
    gamma_arr = design[:, 3]*(gamma_hi-gamma_lo)+gamma_lo

    cosmo = cosmology.setCosmology(cosmo)

    start = time()
    results = np.zeros((ntrain, ndim+npts)).astype('f4')
    gen = zip(range(ntrain), rs_arr, alpha_arr, beta_arr, gamma_arr)
    for i, rs, alpha, beta, gamma in gen:
        if np.mod(i, 100) == 0:
            print("...working on i = {0}".format(i))

        prof = profile_dk14.DK14Profile(rhos=rhos, rs=rs, rt=rt,
            alpha=alpha, beta=beta, gamma=gamma, z=0, mdef='vir')
        total_enclosed = prof.enclosedMassInner(rt*r_by_rt[-1])
        frac_mass_encl = prof.enclosedMassInner(rt*r_by_rt, accuracy=1e-3)/total_enclosed
        log_frac_mass_encl = np.log10(frac_mass_encl)
        results[i, :] = np.concatenate(([rs, alpha, beta, gamma], log_frac_mass_encl))
    end = time()
    runtime = end-start
    msg = "Total runtime = {0:.2f} seconds for {1} models = {2:.2f} s/model"
    print(msg.format(runtime, ntrain, runtime/ntrain))
    np.save(outname, results)
    np.save('DATA/r_by_rt', r_by_rt)
