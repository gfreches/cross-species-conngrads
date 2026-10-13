"""Script 1: average the subject connectivity blueprints of one species.

Each subject blueprint is a .dscalar.nii with one map per tract (in the order of common.TRACTS) over the surface
vertices of one hemisphere. The mean over subjects is renormalised so that every vertex sums to one across tracts
and written to results/1_average_blueprints/average_<species>_blueprint.<H>.func.gii.

The subject blueprints are not part of this repository; the averages are.

Usage (from the project root):
    python code/1_average_blueprints.py --species human --subjects human_subjects.txt \\
        --pattern "/data/human/{subject}_bp_midthickness_inv_{hemisphere}/BP.{hemisphere}.dscalar.nii"
"""
import argparse
import os

import nibabel as nib
import numpy as np

import common as c


def average(files):
    """Mean of the subject blueprints (tracts x vertices), renormalised to sum to one at every vertex."""
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} of {len(files)} blueprints are missing, e.g. {missing[0]}')
    mean = np.mean([nib.load(f).get_fdata() for f in files], axis=0)
    total = mean.sum(0)
    return np.divide(mean, total, out=np.zeros_like(mean), where=total > 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--species', required=True, help='name used in the output file, e.g. human')
    parser.add_argument('--subjects', required=True, help='text file with one subject ID per line')
    parser.add_argument('--pattern', required=True, help='path of a subject blueprint, with {subject} and {hemisphere}')
    parser.add_argument('--hemispheres', default='L,R')
    args = parser.parse_args()

    with open(args.subjects) as f:
        subjects = [s.strip() for s in f if s.strip()]
    for h in args.hemispheres.split(','):
        blueprint = average([args.pattern.format(subject=s, hemisphere=h) for s in subjects])
        if len(blueprint) != len(c.TRACTS):
            raise ValueError(f'expected {len(c.TRACTS)} tract maps, found {len(blueprint)}')
        path = f'{c.ROOT}/' + c.BLUEPRINT.format(species=args.species, h=h)
        c.write_gifti(path, blueprint, h, c.TRACTS)
        print(f'{args.species} {h}: average of {len(subjects)} subjects written to {path}')
