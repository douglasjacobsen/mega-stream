!
!
! Copyright 2017 Tom Deakin, University of Bristol
!
! This file is part of mega-stream.
!
! mega-stream is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! mega-stream is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with mega-stream.  If not, see <http://www.gnu.org/licenses/>.
!
! This aims to investigate the limiting factor for a simple kernel, in particular
! where bandwidth limits not to be reached, and latency becomes a dominating factor.


! Sweep kernel
subroutine sweeper(rank,lrank,rrank,            &
                   nang,ang_set,nang_sets,      &
                   nx,ny,ng,nsweeps,chunk,      &
                   aflux0,aflux1,sflux,         &
                   psii,psij,                   &
                   mu,eta,                      &
                   w,v,dx,dy,                   &
                   buf)

  use comms

  implicit none

  integer :: rank, lrank, rrank
  integer :: nang, ang_set, nang_sets ! ang_set is the size of an angle set, nang_sets is the number of such sets
  integer :: nx, ny, ng, nsweeps, chunk
  real(kind=8) :: aflux0(ang_set,nx,ny,nang_sets,nsweeps,ng)
  real(kind=8) :: aflux1(ang_set,nx,ny,nang_sets,nsweeps,ng)
  real(kind=8) :: sflux(nx,ny,ng)
  real(kind=8) :: psii(ang_set,chunk,nang_sets,ng)
  real(kind=8) :: psij(ang_set,nx,nang_sets,ng)
  real(kind=8) :: mu(ang_set,nang_sets)
  real(kind=8) :: eta(ang_set,nang_sets)
  real(kind=8) :: w(ang_set,nang_sets)
  real(kind=8) :: v
  real(kind=8) :: dx, dy
  real(kind=8) :: buf(ang_set,chunk,nang_sets,ng)

  integer :: a, as, i, j, g, c, cj, sweep
  integer :: istep, jstep ! Spatial step direction
  integer :: xmin, xmax   ! x-dimension loop bounds
  integer :: ymin, ymax   ! y-dimension (chunk) loop bounds
  integer :: cmin, cmax   ! Chunk loop bounds
  integer :: nchunks
  real(kind=8) :: psi

  ! Calculate number of chunks in y-dimension
  nchunks = ny / chunk

  do sweep = 1, nsweeps
    ! Set sweep directions
    select case (sweep)
      case (1)
        istep = -1
        xmin = nx
        xmax = 1
        jstep = -1
        ymin = chunk
        ymax = 1
        cmin = nchunks
        cmax = 1
      case (2)
        istep = -1
        xmin = nx
        xmax = 1
        jstep = 1
        ymin = 1
        ymax = chunk
        cmin = 1
        cmax = nchunks
      case (3)
        istep = 1
        xmin = 1
        xmax = nx
        jstep = -1
        ymin = chunk
        ymax = 1
        cmin = nchunks
        cmax = 1
      case (4)
        istep = 1
        xmin = 1
        xmax = nx
        jstep = 1
        ymin = 1
        ymax = chunk
        cmin = 1
        cmax = nchunks
    end select

    ! Zero boundary data every sweep
    psii = 0.0_8
    psij = 0.0_8

    do c = cmin, cmax, jstep ! Loop over chunks

      ! Recv y boundary data for chunk
      psii = 0.0_8
      if (istep .eq. 1) then
        call recv(psii, ang_set*chunk*nang_sets*ng, lrank)
      else
        call recv(psii, ang_set*chunk*nang_sets*ng, rrank)
      end if

      !$omp parallel do private(cj,j,i,a,psi)
      do g = 1, ng                 ! Loop over energy groups
        do cj = ymin, ymax, jstep  ! Loop over cells in chunk (y-dimension)
          ! Calculate y index with respect to ny
          j = (c-1)*chunk + cj
          do i = xmin, xmax, istep ! Loop over x-dimension
!dir$ vector nontemporal(aflux1)
            do as = 1, nang_sets   ! Loop over angle sets
            do a = 1, ang_set      ! Loop over angles in set
              ! Calculate angular flux
              psi = (mu(a,as)*psii(a,cj,as,g) + eta(a,as)*psij(a,i,as,g) + v*aflux0(a,i,j,as,sweep,g)) &
                    / (0.07_8 + 2.0_8*mu(a,as)/dx + 2.0_8*eta(a,as)/dy + v)

              ! Outgoing diamond difference
              psii(a,cj,as,g) = 2.0_8*psi - psii(a,cj,as,g)
              psij(a,i,as,g) = 2.0_8*psi - psij(a,i,as,g)
              aflux1(a,i,j,as,sweep,g) = 2.0_8*psi - aflux0(a,i,j,as,sweep,g)
  
              ! Reduction
              sflux(i,j,g) = sflux(i,j,g) + psi*w(a,as)

            end do ! angles in set loop
            end do ! angle sets loop
          end do ! x loop
        end do ! y chunk loop
      end do ! group loop
      !$omp end parallel do

      ! Send y boundary data for chunk
      ! NB non-blocking so need to buffer psii, making sure previous send has finished
      call wait_on_sends
      buf = psii
      if (istep .eq. 1) then
        call send(buf, ang_set*chunk*nang_sets*ng, rrank)
      else
        call send(buf, ang_set*chunk*nang_sets*ng, lrank)
      end if

    end do ! chunk loop
  end do ! sweep loop

end subroutine sweeper

