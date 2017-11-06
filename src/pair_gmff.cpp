/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Kurt Smith (U Pittsburgh)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "pair_gmff.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10

/* ---------------------------------------------------------------------- */

PairGMFF::PairGMFF(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  random = NULL;
}

/* ---------------------------------------------------------------------- */

PairGMFF::~PairGMFF()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(gamma);
    memory->destroy(sigma);
    memory->destroy(a0); // added lines
    memory->destroy(a1);
    memory->destroy(a2);
    memory->destroy(a4);
    memory->destroy(a6);
    memory->destroy(a8);
    memory->destroy(a10);
  }

  if (random) delete random;
}

/* ---------------------------------------------------------------------- */

void PairGMFF::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,dot,wd,randnum,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
//        if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
        rinv = 1.0/r;
        delvx = vxtmp - v[j][0];
        delvy = vytmp - v[j][1];
        delvz = vztmp - v[j][2];
        dot = delx*delvx + dely*delvy + delz*delvz;
        wd = 1.0 - r/cut[itype][jtype];
        randnum = random->gaussian();

        // conservative force = a0 * wd
        // drag force = -gamma * wd^2 * (delx dot delv) / r
        // random force = sigma * wd * rnd * dtinvsqrt;

        fpair = a0[itype][jtype] + 
            a1[itype][jtype] * wd +
            a2[itype][jtype] * pow(wd, 2) +
            a4[itype][jtype] * pow(wd, 4) +
            a6[itype][jtype] * pow(wd, 6) +
            a8[itype][jtype] * pow(wd, 8) +
            a10[itype][jtype] * pow(wd, 10);
        fpair -= gamma[itype][jtype]*wd*wd*dot*rinv;
        fpair += sigma[itype][jtype]*wd*randnum*dtinvsqrt;
        fpair *= factor_dpd*rinv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
//          evdwl = 0.5*a0[itype][jtype]*cut[itype][jtype] * wd*wd;
          evdwl = a0[itype][jtype]*cut[itype][jtype] * wd; // added lines
          evdwl += a1[itype][jtype]*cut[itype][jtype] * pow(wd, 2) / 2.0;
          evdwl += a2[itype][jtype]*cut[itype][jtype] * pow(wd, 3) / 3.0;
          evdwl += a4[itype][jtype]*cut[itype][jtype] * pow(wd, 5) / 5.0;
          evdwl += a6[itype][jtype]*cut[itype][jtype] * pow(wd, 7) / 7.0;
          evdwl += a8[itype][jtype]*cut[itype][jtype] * pow(wd, 9) / 9.0;
          evdwl += a10[itype][jtype]*cut[itype][jtype] * pow(wd, 11) / 11.0;
          evdwl *= factor_dpd;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz); // WHATS THIS?
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGMFF::allocate()
{
  int i,j;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(gamma,n+1,n+1,"pair:gamma");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(a0,n+1,n+1,"pair:a0"); // added lines
  memory->create(a1,n+1,n+1,"pair:a1");
  memory->create(a2,n+1,n+1,"pair:a2");
  memory->create(a4,n+1,n+1,"pair:a4");
  memory->create(a6,n+1,n+1,"pair:a6");
  memory->create(a8,n+1,n+1,"pair:a8");
  memory->create(a10,n+1,n+1,"pair:a10");
  for (i = 0; i <= atom->ntypes; i++)
    for (j = 0; j <= atom->ntypes; j++)
      sigma[i][j] = gamma[i][j] = 0.0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGMFF::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  temperature = force->numeric(FLERR,arg[0]);
  cut_global = force->numeric(FLERR,arg[1]);
  seed = force->inumeric(FLERR,arg[2]);

  // initialize Marsaglia RNG with processor-unique seed

  if (seed <= 0) error->all(FLERR,"Illegal pair_style command");
  delete random;
  random = new RanMars(lmp,seed + comm->me);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGMFF::coeff(int narg, char **arg)
{
  int narg_min = 10, narg_max = 11;   // 2 types, 7 coeffs, gamma, [cut]
  if (narg < narg_min || narg > narg_max)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double a0_one = force->numeric(FLERR,arg[2]);       // added lines
  double a1_one = force->numeric(FLERR,arg[3]);
  double a2_one = force->numeric(FLERR,arg[4]);
  double a4_one = force->numeric(FLERR,arg[5]);
  double a6_one = force->numeric(FLERR,arg[6]);
  double a8_one = force->numeric(FLERR,arg[7]);
  double a10_one = force->numeric(FLERR,arg[8]);
  double gamma_one = force->numeric(FLERR,arg[9]);

  double cut_one = cut_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[10]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a0[i][j] = a0_one;     // added lines
      a1[i][j] = a1_one;
      a2[i][j] = a2_one;
      a4[i][j] = a4_one;
      a6[i][j] = a6_one;
      a8[i][j] = a8_one;
      a10[i][j] = a10_one;
      gamma[i][j] = gamma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGMFF::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair dpd requires ghost atoms store velocity");

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers

  if (force->newton_pair == 0 && comm->me == 0) error->warning(FLERR,
      "Pair dpd needs newton pair on for momentum conservation");

  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGMFF::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  sigma[i][j] = sqrt(2.0*force->boltz*temperature*gamma[i][j]);

  cut[j][i] = cut[i][j];
  a0[j][i] = a0[i][j]; // added lines
  a1[j][i] = a1[i][j];
  a2[j][i] = a2[i][j];
  a4[j][i] = a4[i][j];
  a6[j][i] = a6[i][j];
  a8[j][i] = a8[i][j];
  a10[j][i] = a10[i][j];
  gamma[j][i] = gamma[i][j];
  sigma[j][i] = sigma[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGMFF::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a0[i][j],sizeof(double),1,fp);  // added lines
        fwrite(&a1[i][j],sizeof(double),1,fp);
        fwrite(&a2[i][j],sizeof(double),1,fp);
        fwrite(&a4[i][j],sizeof(double),1,fp);
        fwrite(&a6[i][j],sizeof(double),1,fp);
        fwrite(&a8[i][j],sizeof(double),1,fp);
        fwrite(&a10[i][j],sizeof(double),1,fp);
        fwrite(&gamma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGMFF::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&a0[i][j],sizeof(double),1,fp);  // added lines
          fread(&a1[i][j],sizeof(double),1,fp);
          fread(&a2[i][j],sizeof(double),1,fp);
          fread(&a4[i][j],sizeof(double),1,fp);
          fread(&a6[i][j],sizeof(double),1,fp);
          fread(&a8[i][j],sizeof(double),1,fp);
          fread(&a10[i][j],sizeof(double),1,fp);
          fread(&gamma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&a0[i][j],1,MPI_DOUBLE,0,world);  // added lines
        MPI_Bcast(&a1[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&a2[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&a4[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&a6[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&a8[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&a10[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGMFF::write_restart_settings(FILE *fp)
{
  fwrite(&temperature,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&seed,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGMFF::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&temperature,sizeof(double),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&seed,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&temperature,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);

  // initialize Marsaglia RNG with processor-unique seed
  // same seed that pair_style command initially specified

  if (random) delete random;
  random = new RanMars(lmp,seed + comm->me);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairGMFF::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",
            i,a0[i][i],a1[i][i],a2[i][i],a4[i][i],
            a6[i][i],a8[i][i],a10[i][i],gamma[i][i]); // Added
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairGMFF::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g %g %g %g\n",
              i,j,a0[i][j],a1[i][i],a2[i][i],a4[i][i],a6[i][i],
              a8[i][i],a10[i][i],gamma[i][j],cut[i][j]); // Added
}

/* ---------------------------------------------------------------------- */

double PairGMFF::single(int i, int j, int itype, int jtype, double rsq,
                       double factor_coul, double factor_dpd, double &fforce)
{
  double r,rinv,wd,phi;

  r = sqrt(rsq);   // Deleted EPSILON check
  rinv = 1.0/r;
  wd = 1.0 - r/cut[itype][jtype];

  fforce =  a0[itype][jtype] * factor_dpd*rinv; // added lines
  fforce += a1[itype][jtype]*wd * factor_dpd*rinv;
  fforce += a2[itype][jtype]*pow(wd, 2) * factor_dpd*rinv;
  fforce += a4[itype][jtype]*pow(wd, 4) * factor_dpd*rinv;
  fforce += a6[itype][jtype]*pow(wd, 6) * factor_dpd*rinv;
  fforce += a8[itype][jtype]*pow(wd, 8) * factor_dpd*rinv;
  fforce += a10[itype][jtype]*pow(wd, 10) * factor_dpd*rinv;

  phi = a0[itype][jtype]*cut[itype][jtype] * wd; // added lines
  phi += a1[itype][jtype]*cut[itype][jtype] * pow(wd, 2) / 2.0;
  phi += a2[itype][jtype]*cut[itype][jtype] * pow(wd, 3) / 3.0;
  phi += a4[itype][jtype]*cut[itype][jtype] * pow(wd, 5) / 5.0;
  phi += a6[itype][jtype]*cut[itype][jtype] * pow(wd, 7) / 7.0;
  phi += a8[itype][jtype]*cut[itype][jtype] * pow(wd, 9) / 9.0;
  phi += a10[itype][jtype]*cut[itype][jtype] * pow(wd, 11) / 11.0;
  return factor_dpd*phi;
}
