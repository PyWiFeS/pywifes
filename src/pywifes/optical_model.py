# Copyright (c) 2012 by RSAA Computing Section
#
# WIFES PROJECT
#
# FILENAME optical_model.py
#
# GENERAL DESCRIPTION
#   Code to support the fitting of the optical model
#

from __future__ import division, print_function
from astropy.io import fits as pf
import functools
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pickle

# The number of fitted parameters
nparams = 18

# The coefficients to the Sellmeier equation for BSL7Y glass
A1 = 1.13329383
A2 = 1.36897201e-1
A3 = 7.03456004e-1
B1 = 6.69407868e-3
B2 = 2.37391760e-2
B3 = 7.07030316e1

# The refractive index of air
nAir = 1.000293


def saveData(fname, grating, params, lines, meta):
    """ Save the grating, parameters, and lines of data in pickle file"""
    pickle.dump((grating, params, lines, meta), open(fname, "wb"))


def loadData(fname):
    """ Load the grating, parameters, and lines of data from a file"""
    (grating, params, lines, meta) = pickle.load(open(fname, "r"))
    return grating, params, lines, meta


def saveResamplingData(fname, yrange, grating, bin_x, bin_y, pl, halfframe=False, taros=False, keywlist=None):
    """ Saves the resampling data as a FITS image with one extension for x
    and one for y."""

    if halfframe:
        if taros:
            first = 1
            nslits = 12
        else:
            first = 7
            nslits = 13
    else:
        first = 1
        nslits = 25

    # The length in x (note that for mjc's interpolation we need to use 4096, but for my C++ code
    # we need 4097)
    xlen = int(math.ceil(4096 / bin_x))

    # The range of x values (the same for each row)
    xrnge = np.arange(xlen)

    # The beginnings of a FITS file
    pri = pf.PrimaryHDU(header=None, data=None)
    f = pf.HDUList([pri])

    for s in range(nslits):
        y0, y1 = yrange[s]
        xout = -1 * np.ones((abs(y1 - y0), xlen))
        for y in range(y0, y1):
            # The wavelength solution was not applied at the good location because of
            # a shift along the y-indices (mismatch between the python [0,1,...] and
            # FITS [1,2,3, ....] way of referencing array elements ?)
            xout[y - y0, :] = fitfunc(grating, pl[:nparams], pl[nparams:],
                                      (s + first) * np.ones_like(xrnge),
                                      (y + 1) * bin_y * np.ones_like(xrnge), xrnge)

        xhdu = pf.ImageHDU(header=None, data=xout)
        f.append(xhdu)
    f.update_extend()
    f[0].header.set("PYWWAVEM", "OpticalModel", "PyWiFeS: method for wavelength solution")
    if keywlist is not None:
        for kw, val, desc in keywlist:
            f[0].header.set(kw, val, desc)
    f.writeto(fname, overwrite=True)


# new function from Mike to evaluate optical model
def evaluate_optical_model(x, y, s, grating, bin_x, bin_y, params):
    return fitfunc(grating,
                   params[:nparams],
                   params[nparams:],
                   s,
                   y * bin_y,
                   x)


def plotLines(title, allx, ally, save_fn=None):
    """ Do a plot of the lines used """
    plt.figure(1)
    plt.plot(allx, ally, 'r.', markeredgecolor='w')
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    plt.title(title)
    plt.savefig(save_fn, dpi=300)
    plt.close()


def plotFunc(title, allx, ally, allarcs, f):
    """ Do a plot of the fit"""
    plt.figure()
    plt.plot(allx, f, 'r.', markeredgecolor='w')
    plt.plot(allx, allarcs, 'b.', markeredgecolor='w')
    plt.xlabel("X-axis [pixel]")
    plt.ylabel(r"Wavelength [$\AA$]")
    plt.title(title)
    return


def plotResidKeep(allx, ally, allarcs, resid, keepargs):
    """ Plot the residuals that are being kept, and those that are being
    automatically discarded """
    loseargs = np.logical_not(keepargs)
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(allx[loseargs], resid[loseargs], 'r.')
    plt.plot(allx[keepargs], resid[keepargs], 'b.')
    plt.xlabel("x pixel")
    plt.ylabel(u"data-model \u00C5")
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(resid[loseargs], ally[loseargs], 'r.')
    plt.plot(resid[keepargs], ally[keepargs], 'b.')
    plt.xlabel(u"data=model \u00C5")
    plt.ylabel("y pixel")
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(allarcs[loseargs], resid[loseargs], 'r.')
    plt.plot(allarcs[keepargs], resid[keepargs], 'b.')
    plt.xlabel(u"wavelength \u00C5")
    plt.ylabel(u"data-model \u00C5")
    plt.grid(True)
    plt.title("Lose(Red) / Keep(Blue)")
    plt.show()


def plotResid(title, allx, ally, allarcs, resid, save_fn=None):
    """ Plot the residuals against x, y, and wavelength """
    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(allx, resid, 'r.', markeredgecolor='w')
    plt.xlabel("x pixel")
    plt.ylabel(u"data-model \u00C5")
    plt.grid(True)
    plt.title(title)

    plt.subplot(3, 1, 2)
    plt.plot(resid, ally, 'r.', markeredgecolor='w')
    plt.xlabel(u"data-model \u00C5")
    plt.ylabel("y pixel")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(allarcs, resid, 'r.', markeredgecolor='w')
    plt.xlabel(u"wavelength \u00C5")
    plt.ylabel(u"data-model \u00C5")
    plt.grid(True)

    plt.savefig(save_fn, dpi=300)
    plt.close('all')


def mpfitfunc(p, fjac=None, s=None, y=None, x=None, grating=None, arc=None, err=None):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    model = fitfunc(grating, p[:nparams], p[nparams:], s, y, x)
    # Non-negative status value means MPFIT should continue, negative means
    # stop the
    status = 0
    return [status, (arc - model) / err]


def mperrfunc_alphap(alphap, arg):
    g, s, y, x, a, p, err = arg
    m = fitfunc(g, p, alphap, s, y, x)
    return (a - m) / err


def mpfitfunc_alphap(alphap, fjac=None, alls=None, ally=None, allx=None, gratings=None,
                     allarc=None, allp=None, allerr=None, verbose=False):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    n_cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(int(math.ceil(n_cpus / 2)))
    if verbose:
        print('p=', p)
    partial_mperrfunc = functools.partial(mperrfunc_alphap, alphap)
    out = p.map(partial_mperrfunc, list(zip(gratings, alls, ally, allx, allarc, allp, allerr)))
    p.close()

    # Non-negative status value means MPFIT should continue, negative means
    # stop the fit
    status = 0
    return [status, np.concatenate(out)]


def defaultParams(grating):
    """ Return the default set of parameters """

    if grating == 'u7000':
        d0 = 1948.
        lambda0 = 3850
    elif grating == 'b7000':
        d0 = 1530.
        lambda0 = 4900
    elif grating == 'r7000':
        d0 = 1210.
        lambda0 = 6200
    elif grating == 'i7000':
        d0 = 937.
        lambda0 = 8000
    elif grating == 'r3000':
        d0 = 398.
        lambda0 = 7420
    elif grating == 'b3000':
        d0 = 708.
        lambda0 = 4680

    # Set up the initial set of parameters
    plorig = np.zeros((nparams))

    # Lines per mm
    plorig[0] = d0

    # Input alpha
    plorig[1] = math.radians(22)

    # Phi
    plorig[2] = 0.0

    # Optical centre in x
    plorig[3] = 2048

    # Optical centre in y
    plorig[4] = 2048

    # Radial distortion terms
    plorig[5] = 0.0
    plorig[6] = 0.0
    plorig[7] = 0.0

    # Focal length of camera
    plorig[8] = 261

    # Tilt of detector in x
    plorig[9] = 0.0

    # Tilt of detector in y
    plorig[10] = 0.0

    # Position on detector at which beta0 lands for the central slitlet
    plorig[11] = 2048

    # Position on detector at which gamma=0 lands
    plorig[12] = 2048

    # lambda0
    plorig[13] = lambda0

    if grating == 'r3000':
        plorig[14] = math.radians(22.45)
        plorig[15] = math.radians(20.52)
    elif grating == 'b3000':
        plorig[14] = math.radians(20.61)
        plorig[15] = math.radians(18.64)

    # Curvature coefficients of focal plane
    plorig[16] = 0.0
    plorig[17] = 0.0

    return plorig


def printParams(p, alphap):
    # Extract parameters (and ignore any extras we might be given)
    (d0, input_alpha, phi, xoc, yoc, r1, r2, r3, fcamera, theta_x, theta_y, xdc, ydc, lambda0, Afront, Aback, rx, ry) = p[:nparams]
    print('d0=', d0)
    print('input_alpha=', input_alpha, '(', math.degrees(input_alpha), ' degrees)')
    print('phi=', phi, '(', math.degrees(phi), ' degrees)')
    print('xoc=', xoc)
    print('yoc=', yoc)
    print('r1=', r1)
    print('r2=', r2)
    print('r3=', r3)
    print('fcamera=', fcamera)
    print('theta_x=', theta_x, '(', math.degrees(theta_x), ' degrees)')
    print('theta_y=', theta_y, '(', math.degrees(theta_y), ' degrees)')
    print('xdc=', xdc)
    print('ydc=', ydc)
    print('lambda0=', lambda0)
    print('Afront=', Afront, '(', math.degrees(Afront), ' degrees)')
    print('Aback=', Aback, '(', math.degrees(Aback), ' degrees)')
    print('rx=', rx)
    print('ry=', ry)
    print('alphap=', alphap)


def norm_vector(x):
    norm = np.sum(x**2, axis=-1)**(1. / 2)
    return x / norm[:, np.newaxis]


def snell(n1, n2, norm, light):
    # Calculate the refraction of the light vector moving from medium with refractive
    # index n1 into medium with refractive index n2.  The interface between the media
    # is defined by the vector normal to the surface.
    nratio = n1 / n2
    costheta1 = np.inner(norm, -light)
    sintheta12 = nratio**2 * (1 - costheta1**2)

    if (np.isscalar(nratio)):
        a = nratio * light
    else:
        a = nratio[:, np.newaxis] * light

    # When we would get internal reflection we cheat by making the light continue straight on.
    # There are no valid solutions in which this happens, so it seems like an ok compromise.
    badargs = ((1 - sintheta12) < 0)
    if (not np.isscalar(badargs)):
        sintheta12[badargs] = 1
    elif badargs:
        sintheta12 = 1

    if (np.isscalar(nratio)):
        return a + (nratio * costheta1 - np.sqrt(1 - sintheta12))[np.newaxis].T * norm

    return a + (nratio * costheta1 - np.sqrt(1 - sintheta12))[:, np.newaxis] * norm


def fitfunc(grating, p, alphap, s, y, x):
    # The function that produces lambda given the list of parameters in p and alphap, the list of
    # slitlets in s, and the y and x values of each data point.

    # Test for empty inputs
    if (s.size == 0) or (y.size == 0) or (x.size == 0):
        return np.array([])

    # Flip x for the blue gratings
    if grating in ['u7000', 'b7000', 'b3000']:
        x = 4096 - x

    # Extract parameters (and ignore any extras we might be given)
    (d0, input_alpha, phi, xoc, yoc, r1, r2, r3, fcamera, theta_x, theta_y, xdc, ydc, lambda0, Afront, Aback, rx, ry) = p[:nparams]

    # Calculate these now for later use
    sinphi = math.sin(phi / 2)
    cosphi = math.cos(phi / 2)

    # Angstroms per line
    a0 = 1.0e7 / d0

    # Constant pixel size
    pix2mm = 15e-3  # + pix2mmoff # mm

    # X and Y in mm on the detector
    xd = pix2mm * x
    yd = pix2mm * y

    # The centre of radial distortion in mm
    xoc *= pix2mm
    yoc *= pix2mm

    # Account for radial distortion in x
    r = np.sqrt((xd - xoc)**2 + (yd - yoc)**2)
    xu = xd + (xd - xoc) * (r1 * r**2 + r2 * r**4 + r3 * r**6)
    yu = yd

    # The position in x on the detector where beta0 lands for the central slitlet
    xdc *= pix2mm

    # The position in y on the detector where gamma=0 lands
    ydc *= pix2mm

    # Distance in x and y from the position on the detector where
    # beta=beta0 FOR THE CENTRAL SLITLET, and where gamma=0
    x0 = xu - xdc
    y0 = yu - ydc

    # A simple adjustment for a focal plane curved in x
    z0 = rx * x0**2 + ry * y0**2

    # Set up for the rotation of the detector
    sx = math.sin(theta_x)
    cx = math.cos(theta_x)
    sy = math.sin(theta_y)
    cy = math.cos(theta_y)

    # Rotation by theta_y then theta_x
    rot_matrix = np.matrix([[cx, sx * sy, sx * cy], [0, cy, -sy], [-sx, cx * sy, cx * cy]])
    rot_coords = (np.matrix([x0, y0, z0]).T * rot_matrix).getA()

    # Our coordinate system now has z in the direction of the incoming ray where beta=beta0
    # and gamma=0, and x and y in the plane that is orthogonal to that ray where x is in the
    # dispersal direction and y is in the off-axis direction.

    # Now translate so that the origin is at the focus
    rot_coords[:, 2] += fcamera

    # Now rotate again by gamma=0, then beta0 to bring us to the coordinate system that is aligned
    # with the grating.  Firste we need to work out what beta0 is though.

    if grating[1:] == '7000':
        # The off-axis angle at y=ydc is defined to be gamma=0
        # beta0 is the angle of refraction for alpha0 (and remember that it lands
        # on the detector at x=xdc in the central slitlet)
        alpha0 = input_alpha
        tmpsin = np.clip((lambda0 / a0) - math.sin(alpha0 + phi / 2), -1.0, 1.0)
        beta0 = -(math.asin(tmpsin) + phi / 2)
    else:
        # We construct the vectors normal to the outwards faces
        # of the two prisms
        norm_front = np.array([math.tan(Afront), 0, -1])
        norm_front /= np.linalg.norm(norm_front)

        norm_back = np.array([-math.tan(Aback), 0, -1])
        norm_back /= np.linalg.norm(norm_back)

        # Central wavelength squared
        l02 = (lambda0 / 1e4)**2

        # Refractive index of prisms at central wavelength
        n0 = math.sqrt(1 + (A1 * l02 / (l02 - B1)) + (A2 * l02 / (l02 - B2)) + (A3 * l02 / (l02 - B3)))

        # The angle of incidence for the central slitlet
        prism_in_alpha0 = input_alpha

        # A unit vector in the direction of the incoming light for the central slitlet.
        prism_in_light0 = np.array([math.tan(prism_in_alpha0), 0.0, 1])
        prism_in_light0 /= np.linalg.norm(prism_in_light0)

        # Light exiting the front prism onto the grating
        # for the central wavelength of the central slitlet
        grating_in_light0 = snell(nAir, n0, norm_front, prism_in_light0)

        # The angle of incidence on the grating
        grating_in_alpha0 = math.atan(grating_in_light0[0] / grating_in_light0[2])

        # The angle of diffraction for the central wavelength of the central slitlet.
        # We calculate it using the grating equation
        tmpsin = np.clip((lambda0 / a0) - math.sin(grating_in_alpha0 + phi / 2), -1.0, 1.0)
        grating_out_beta0 = math.asin(tmpsin) + phi / 2

        # Turn the angle into a unit vector in the direction of the light exiting the grating
        grating_out_light0 = np.array([-math.tan(grating_out_beta0), 0.0, 1])
        grating_out_light0 /= np.linalg.norm(grating_out_light0)

        # Now calculate the unit vector in the direction of the light exiting the back prism.
        prism_out_light0 = snell(n0, nAir, norm_back, grating_out_light0)

        # The angle of diffraction as it exits the back prism.
        prism_out_beta0 = math.atan(prism_out_light0[0] / prism_out_light0[2])
        beta0 = prism_out_beta0

    # Set up for the rotation
    sb = math.sin(beta0)
    cb = math.cos(beta0)

    # We use gamma=0.  Fitting for non-zero gamma0 proved to be pointless.
    sg = 0.0
    cg = 1.0

    # Rotation by gamma=0, then beta0
    rot_matrix = np.matrix([[cb, sb * sg, sb * cg], [0, cg, -sg], [-sb, cb * sg, cb * cg]])
    rot_coords = (rot_coords * rot_matrix).getA()

    # These are our coordinates wrt to the focus.  This means we can read beta and gamma
    # straight out of these through simple geometry.
    xc = rot_coords[:, 0]
    yc = rot_coords[:, 1]
    zc = rot_coords[:, 2]

    cx2 = xc**2 + zc**2
    cx = np.sqrt(cx2)
    cy = np.sqrt(cx2 + yc**2)

    # Offset to alpha as a result of the staircase effect
    staircase_offset = (s - 13) * math.atan((15e-3 * 2) / 261)

    # Off axis angle
    cosgamma = cx / cy

    # Do we need to worry about the prisms?
    if grating[1:] == '7000':
        # No, it's a 7000 grating

        # Input angle
        alpha = input_alpha + staircase_offset + alphap[s - 1]
        sinalpha = np.sin(alpha)
        cosalpha = np.cos(alpha)

        # Angle of refraction
        sinbeta = xc / cx
        cosbeta = zc / cx

    else:
        # It's a 3000 grating.  There's some more work to do.

        # A vector of all ones that has the right size
        ones = np.ones_like(s)

        # A basic guess at lambda
        lambda1 = x0 / pix2mm + lambda0

        # An estimate of the refractive index
        l12 = (lambda1 / 1e4)**2
        n12 = 1 + (A1 * l12 / (l12 - B1)) + (A2 * l12 / (l12 - B2)) + (A3 * l12 / (l12 - B3))

        # The refractive index calculation can go wrong when lambda gets way out of range.
        # This only happens for solutions that are not actually sensible, but that's what
        # you get in the middle of using mpfit.  We avoid NaNs by just pretending the problem
        # doesn't exist and putting a default value in for n.
        badargs = (n12 <= 0)
        n12[badargs] = nAir**2
        n1 = np.sqrt(n12)

        # The angles of incidence for all the other slitlets
        prism_in_alpha = input_alpha + staircase_offset + alphap[s - 1]
        tan_prism_in_alpha = np.tan(prism_in_alpha)

        # Unit vectors in the direction of the incoming light for all slitlets.
        prism_in_light = np.column_stack((tan_prism_in_alpha, yc / zc, ones))
        prism_in_light = norm_vector(prism_in_light)

        # Light exiting the front prism onto the grating for all slitlets
        grating_in_light = snell(nAir, n1, norm_front, prism_in_light)

        # The angle of incidence on the grating
        cx = np.sqrt(grating_in_light[:, 0]**2 + grating_in_light[:, 2]**2)
        sinalpha = grating_in_light[:, 0] / cx
        cosalpha = grating_in_light[:, 2] / cx

        # Construct a unit vector in the direction of light exiting the back prism
        prism_out_light = np.column_stack((-rot_coords[:, 0], rot_coords[:, 1], rot_coords[:, 2]))
        prism_out_light = norm_vector(prism_out_light)

        # Now that we have vectors for all the light exiting the back prism, we can work
        # backwards to find out where this light exited the grating.
        # This gives us the angle of diffraction from the grating.
        grating_out_light = -snell(nAir, n1, -norm_back, -prism_out_light)
        cx = np.sqrt(grating_out_light[:, 0]**2 + grating_out_light[:, 2]**2)
        sinbeta = -grating_out_light[:, 0] / cx
        cosbeta = grating_out_light[:, 2] / cx

        # Now we take an iterative approach.  We need to know lambda in order to calculate the
        # refractive index of the prisms.  But we need to calculate the refraction through the
        # prisms in order to calculate lambda.  Our initial estimates are good enough, and the
        # calculations converge quickly.
        finished = False
        count = 0
        maxCount = 7
        while (not finished) and (count < maxCount):
            count += 1
            # Calculate wavelength from the grating equation, using our best guess at alpha, beta, and gamma
            lambdaold = lambda1
            sin_alpha_p_phi_on_2 = sinalpha * cosphi + cosalpha * sinphi
            sin_beta_m_phi_on_2 = sinbeta * cosphi - cosbeta * sinphi
            lambda1 = a0 * (sin_alpha_p_phi_on_2 + sin_beta_m_phi_on_2) * cosgamma
            max_change = np.max(np.abs(lambda1 - lambdaold))

            # If the solution has converged to better than 1 Angstrom then this is our
            # last iteration
            if (max_change < 1):
                finished = True

            # Calculate refractive index for prisms
            l2 = (lambda1 / 1e4)**2
            n2 = 1 + (A1 * l2 / (l2 - B1)) + (A2 * l2 / (l2 - B2)) + (A3 * l2 / (l2 - B3))

            # The refractive index calculation can go wrong when lambda gets way out of range.
            # This only happens for solutions that are not actually sensible, but that's what
            # you get in the middle of using mpfit.  We avoid NaNs by just pretending the problem
            # doesn't exist and putting a default value in for n.
            badargs = (n2 <= 0)
            n2[badargs] = nAir**2
            n = np.sqrt(n2)

            # Unit vectors in the direction of light entering the grating
            grating_in_light = snell(nAir, n, norm_front, prism_in_light)
            cx = np.sqrt(grating_in_light[:, 0]**2 + grating_in_light[:, 2]**2)
            sinalpha = grating_in_light[:, 0] / cx
            cosalpha = grating_in_light[:, 2] / cx

            # The angle at which light exits the grating (calculated backwards from where it exits the back prism)
            grating_out_light = -snell(nAir, n, -norm_back, -prism_out_light)

            # Our best estimate at the angle of diffraction
            cx = np.sqrt(grating_out_light[:, 0]**2 + grating_out_light[:, 2]**2)
            sinbeta = -grating_out_light[:, 0] / cx
            cosbeta = grating_out_light[:, 2] / cx

    sin_alpha_p_phi_on_2 = sinalpha * cosphi + cosalpha * sinphi
    sin_beta_m_phi_on_2 = sinbeta * cosphi - cosbeta * sinphi
    return a0 * (sin_alpha_p_phi_on_2 + sin_beta_m_phi_on_2) * cosgamma


def errfunc(grating, p, alphap, s, y, x, a):
    return a - fitfunc(grating, p, alphap, s, y, x)


def extractArrays(lines, grating, bin_x, bin_y):
    # Extract the s, y, x, and a arrays from the data set
    alls = np.array(lines[:, 0], dtype=int)
    ally = lines[:, 1] * bin_y
    allx = lines[:, 2] * bin_x
    allarcs = lines[:, 3]
    return alls, ally, allx, allarcs


def excludeAuto(lines, grating, bin_x, bin_y, resid, sigma, plot, verbose):
    # Automatically exclude lines that have large residuals
    alls, ally, allx, allarcs = extractArrays(lines, grating, bin_x, bin_y)
    allrms = []
    for s in set(alls):
        args = (alls == s)
        for a in set(allarcs[args]):
            arcargs = np.logical_and(args, (allarcs == a))
            rms = np.max(np.abs(resid[arcargs]))
            allrms.append([s, a, rms])

    allrms = np.asarray(allrms)
    mean = allrms[:, 2].mean()
    std = allrms[:, 2].std()

    keepargs = np.ones_like(alls, dtype=bool)
    for s, a, rms in allrms:
        if (rms > sigma * std + mean):
            if (verbose):
                print('Excluding', s, a, rms)
            excludeargs = np.logical_and(alls == s, allarcs == a)
            keepargs = np.logical_and(keepargs, np.logical_not(excludeargs))

    if (plot):
        plotResidKeep(allx, ally, allarcs, resid, keepargs)

    return lines[keepargs]
