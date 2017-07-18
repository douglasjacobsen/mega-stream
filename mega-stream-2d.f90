!
!
! Copyright 2016 Tom Deakin, University of Bristol
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
!

PROGRAM megastream

  USE iso_c_binding
  USE omp_lib

  IMPLICIT NONE

  ! Use C for aligned allocation
  INTERFACE

    TYPE(C_PTR) FUNCTION ALLOC(len) BIND(C)
      IMPORT :: C_PTR
      IMPLICIT NONE
      INTEGER :: len
    END FUNCTION

    SUBROUTINE ALLOC_FREE(cptr) BIND(C)
      IMPORT :: C_PTR
      IMPLICIT NONE
      TYPE(C_PTR) :: cptr
    END SUBROUTINE

  END INTERFACE

  ! Constant parameters

  ! Arrays are defined in terms of 3 sizes: inner, middle and outer.
  ! The large arrays are of size inner*middle*middle*middle*outer.
  ! The medium arrays are of size inner*middle*middle*outer.
  ! The small arrays are of size inner and are indexed with 1 index.

  INTEGER, PARAMETER :: OUTER = 64
  INTEGER, PARAMETER :: MIDDLE = 16
  INTEGER, PARAMETER :: INNER = 128

  ! Tollerance with which to check final array values
  REAL(8), PARAMETER :: TOLR = 1.0E-15

  ! Vector length if machine-specific
  INTEGER, PARAMETER :: VLEN = 8

  ! Variables

  ! Default strides
  INTEGER :: Ni = INNER
  INTEGER :: Nj = MIDDLE
  INTEGER :: Nk = MIDDLE
  INTEGER :: Nm = OUTER
  INTEGER :: Ng
  INTEGER :: chunk = 1

  ! Number of iterations to run benchmark
  INTEGER :: ntimes = 100

  ! Arrays
  REAL(8), DIMENSION(:,:,:,:,:), POINTER :: q, r, ptr_tmp
  TYPE(C_PTR) :: q_pt, r_pt
  REAL(8), DIMENSION(:,:,:,:), ALLOCATABLE :: x, y
  TYPE(C_PTR) :: x_pt, y_pt
  REAL(8), DIMENSION(:,:), ALLOCATABLE :: a, b
  TYPE(C_PTR) :: a_pt, b_pt
  REAL(8), DIMENSION(:,:,:), ALLOCATABLE :: total
  TYPE(C_PTR) :: total_pt

  REAL(8), DIMENSION(:), ALLOCATABLE :: timings
  REAL(8) :: tick, tock
  REAL(8) :: start, finish

  REAL(8) :: moved

  INTEGER :: i, j, k, m, t

  ! Print information
  WRITE(*, '(a)') 'MEGA-STREAM! - v0.3'
  WRITE(*, *)

  CALL parse_args(Ni, Nj, Nk, Nm, ntimes, chunk)

  WRITE(*, '(a,I,a,f8.1,a)') 'Small arrays: ', Ni, ' elements ', Ni*8*1.0E-3, 'KB'
  WRITE(*, '(a,I4,a,I4,a,I4,a,I4,f8.1,a)') 'Large arrays: ', Ni, ' x', Nj, &
    ' x', Nk, ' x', Nm, Ni*Nj*Nk*Nm*8*1.0E-6, 'MB'
  !WRITE(*, '(a,I,a,I,a,f8.1,a)') 'Medium arrays: ', S_size, 'x', M_size, ' elements', S_size*M_size*8*1.0E-6, 'MB'
  !WRITE(*, '(a,f8.1,a)') 'Memory footprint: ', 8 * 1.0E-6 * ( &
    !2.0*L_size*M_size*S_size + &
    !3.0*M_size*S_size +        &
    !3.0*S_size +               &
    !L_size*M_size)             &
    !, ' MB'
  WRITE(*, '(a,I,a)') 'Running ', ntimes, ' times'
  WRITE(*, *)

  ! Total memory moved
  moved = 8 * 1.0E-6 * ( &
    Ni*Nj*Nk*Nm  +    & ! read q
    Ni*Nj*Nk*Nm  +    & ! write r
    Ni + Ni + Ni    +    & ! read a, b and c
    2.0*Ni*Nj*Nm +    & ! read and write x
    2.0*Ni*Nk*Nm +    & ! read and write y
    2.0*Nj*Nk*Nm )      ! read and write sum

  ! Split inner-most dimension into VLEN-sized chunks
  Ng = Ni / VLEN
  WRITE(*, '(a,I,a,I)') 'Inner dimension split into ', Ng, ' chunks of size ', VLEN

  ! Allocate memory
  q_pt = ALLOC(VLEN*Nj*Nk*Ng*Nm)
  r_pt = ALLOC(VLEN*Nj*Nk*Ng*Nm)
  x_pt = ALLOC(VLEN*Nj*Ng*Nm)
  y_pt = ALLOC(VLEN*Nk*Ng*Nm)
  a_pt = ALLOC(VLEN*Ng)
  b_pt = ALLOC(VLEN*Ng)
  total_pt = ALLOC(Nj*Nk*Nm)

  CALL C_F_POINTER(q_pt, q, (/VLEN,Nj,Nk,Ng,Nm/))
  CALL C_F_POINTER(r_pt, r, (/VLEN,Nj,Nk,Ng,Nm/))
  CALL C_F_POINTER(x_pt, x, (/VLEN,Nj,Ng,Nm/))
  CALL C_F_POINTER(y_pt, y, (/VLEN,Nk,Ng,Nm/))
  CALL C_F_POINTER(a_pt, a, (/VLEN,Ng/))
  CALL C_F_POINTER(b_pt, b, (/VLEN,Ng/))
  CALL C_F_POINTER(total_pt, total, (/Nj,Nk,Nm/))

  CALL init(VLEN, Nj, Nk, Ng, Nm, r, q, x, y, a, b, total)

  ALLOCATE(timings(ntimes))

  start = omp_get_wtime()

  ! Run the kernel multiple times
  DO t = 1, ntimes
    tick = omp_get_wtime()

    CALL kernel(VLEN, Nj, Nk, Ng, Nm, r, q, x, y, a, b, total)

    tock = omp_get_wtime()
    timings(t) = tock-tick

    tock = omp_get_wtime()
    timings(t) = tock-tick

    ! Swap the pointers
    ptr_tmp => q
    q => r
    r => ptr_tmp
    NULLIFY(ptr_tmp)


  END DO ! t

  finish = omp_get_wtime()

  ! Check the results
  WRITE(*, '(a,G)') "Sum total: ", SUM(total)
  WRITE(*, *)

  ! Print timings
  WRITE(*, '(a,a,a,a)') 'Bandwidth MB/s ', 'Min time ', 'Max time ', 'Avg time'
  WRITE(*, '(f12.1,f11.6,f11.6,f11.6)') &
    moved / MINVAL(timings(2:ntimes)), &
    MINVAL(timings(2:ntimes)), &
    MAXVAL(timings(2:ntimes)), &
    SUM(timings(2:ntimes)) / (ntimes - 1)
 WRITE(*, '(a,f11.6)') 'Total time: ', finish-start

! Deallocate memory
  CALL ALLOC_FREE(q_pt)
  CALL ALLOC_FREE(r_pt)
  CALL ALLOC_FREE(x_pt)
  CALL ALLOC_FREE(y_pt)
  CALL ALLOC_FREE(a_pt)
  CALL ALLOC_FREE(b_pt)
  CALL ALLOC_FREE(total_pt)
  DEALLOCATE(timings)

END PROGRAM megastream

!**************************************************************************
!* Kernel
!*************************************************************************/
SUBROUTINE kernel(VLEN, Nj, Nk, Ng, Nm, r, q, x, y, a, b, total)

  IMPLICIT NONE
  
  INTEGER, INTENT(IN) :: VLEN, Nj, Nk, Ng, Nm

  REAL(8), DIMENSION(VLEN, Nj, Nk, Ng, Nm), INTENT(INOUT) :: q, r
  REAL(8), DIMENSION(VLEN, Nj, Ng, Nm), INTENT(INOUT) :: x
  REAL(8), DIMENSION(VLEN, Nk, Ng, Nm), INTENT(INOUT) :: y
  REAL(8), DIMENSION(VLEN, Ng), INTENT(IN) :: a, b
  REAL(8), DIMENSION(Nj, Nk, Nm), INTENT(INOUT) :: total

  ! Local variables
  INTEGER :: v, j, k, g, m
  REAL(8) :: tmp_r, tmp_total

!$OMP PARALLEL DO PRIVATE(tmp_r, tmp_total)
  DO m = 1, Nm
    DO g = 1, Ng
      DO k = 1, Nk
        DO j = 1, Nj
          tmp_total = 0.0_8
          CALL MM_PREFETCH(q(1+32*VLEN,j,k,g,m), 1)
          !DIR$ VECTOR NONTEMPORAL(r)
          !$OMP SIMD REDUCTION(+:tmp_total) ALIGNED(a,b,x,y,r,q:64)
          DO v = 1, VLEN
            ! Set r
            tmp_r = q(v,j,k,g,m) +  &
              a(v,g)*x(v,j,g,m) +     &
              b(v,g)*y(v,k,g,m)

            ! Update x, y and z
            x(v,j,g,m) = 0.2_8*tmp_r - x(v,j,g,m)
            y(v,k,g,m) = 0.2_8*tmp_r - y(v,k,g,m)

            ! Reduce over Ni
            tmp_total = tmp_total + tmp_r

            ! Save r
            r(v,j,k,g,m) = tmp_r

          END DO ! i

          total(j,k,m) = total(j,k,m) + tmp_total

        END DO ! j
      END DO ! k
    END DO ! g
  END DO ! m
!$OMP END PARALLEL DO

END SUBROUTINE kernel
!**************************************************************************
!* End of Kernel
!*************************************************************************/

! Initilise the arrays
SUBROUTINE init(VLEN, Nj, Nk, Ng, Nm, r, q, x, y, a, b, total)

  IMPLICIT NONE

  INTEGER, INTENT(IN) :: VLEN, Nj, Nk, Ng, Nm
  REAL(8), DIMENSION(VLEN, Nj, Nk, Ng, Nm), INTENT(INOUT) :: q, r
  REAL(8), DIMENSION(VLEN, Nj, Ng, Nm), INTENT(INOUT) :: x
  REAL(8), DIMENSION(VLEN, Nk, Ng, Nm), INTENT(INOUT) :: y
  REAL(8), DIMENSION(VLEN, Ng), INTENT(INOUT) :: a, b
  REAL(8), DIMENSION(Nj, Nk, Nm), INTENT(INOUT) :: total

  ! Starting values
  REAL(8), PARAMETER :: R_START = 0.0_8
  REAL(8), PARAMETER :: Q_START = 0.01_8
  REAL(8), PARAMETER :: X_START = 0.02_8
  REAL(8), PARAMETER :: Y_START = 0.03_8
  REAL(8), PARAMETER :: A_START = 0.06_8
  REAL(8), PARAMETER :: B_START = 0.07_8

  INTEGER :: v, j, k, g, m

!$OMP PARALLEL
  ! q and r
!$OMP DO
  DO m = 1, Nm
    DO g = 1, Ng
      DO k = 1, Nk
        DO j = 1, Nj
          DO v = 1, VLEN
            r(v,j,k,g,m) = R_START
            q(v,j,k,g,m) = Q_START
          END DO
        END DO
       END DO
    END DO
  END DO
!$OMP END DO

  ! x
!$OMP DO
  DO m = 1, Nm
    DO g = 1, Ng
      DO j = 1, Nj
        DO v = 1, VLEN
          x(v,j,g,m) = X_START
        END DO
      END DO
    END DO
  END DO
!$OMP END DO

  ! y
!$OMP DO
  DO m = 1, Nm
    DO g = 1, Ng
      DO k = 1, Nk
        DO v = 1, VLEN
          y(v,k,g,m) = Y_START
        END DO
      END DO
    END DO
  END DO
!$OMP END DO

  ! a, b
!$OMP DO
  DO g = 1, Ng
    DO v = 1, VLEN
      a(v,g) = A_START
      b(v,g) = B_START
    END DO
  END DO
!$OMP END DO

  ! sum
!$OMP DO
  DO m = 1, Nm
    DO k = 1, Nk
      DO j = 1, Nj
        total(j,k,m) = 0.0_8
      END DO
    END DO
  END DO
!$OMP END DO

!$OMP END PARALLEL
END SUBROUTINE

SUBROUTINE parse_args(Ni, Nj, Nk, Nm, ntimes, chunk)

  IMPLICIT NONE

  INTEGER, INTENT(INOUT) :: Ni, Nj, Nk, Nm, ntimes, chunk

  CHARACTER(len=32) :: arg

  INTEGER :: i = 1

  DO WHILE (i <= iargc())
    CALL getarg(i, arg)
    IF (arg .EQ. "--outer") THEN
      i = i + 1
      CALL getarg(i, arg)
      READ(arg, *) Nm
    ELSE IF (arg .EQ. "--inner") THEN
      i = i + 1
      CALL getarg(i, arg)
      READ(arg, *) Ni
    ELSE IF (arg .EQ. "--Nk") THEN
      i = i + 1
      CALL getarg(i, arg)
      READ(arg, *) Nk
    ELSE IF (arg .EQ. "--Nj") THEN
      i = i + 1
      CALL getarg(i, arg)
      READ(arg, *) Nj
    ELSE IF (arg .EQ. "--chunk") THEN
      i = i + 1
      CALL getarg(i, arg)
      READ(arg, *) chunk
    ELSE IF (arg .EQ. "--ntimes") THEN
      i = i + 1
      CALL getarg(i, arg)
      READ(arg, *) ntimes
      IF (ntimes < 2) THEN
        WRITE(*, *) "ntimes must be 2 or greater"
        WRITE(*, *)
        STOP
      END IF
    ELSE IF (arg .EQ. "--help") THEN
      WRITE(*, *) "--outer  n  Set size of outer dimension"
      WRITE(*, *) "--inner  n  Set size of inner dimension"
      WRITE(*, *) "--Nj     n  Set size of the j dimension"
      WRITE(*, *) "--Nk     n  Set size of the k dimension"
      WRITE(*, *) "--chunk  n  Set size of chunk in k dimension"
      WRITE(*, *) "--ntimes n  Run the benchmark n times"
      WRITE(*, *)
      STOP
    ELSE
      WRITE(*, *) "Unrecognised argument ", arg
      WRITE(*, *)
      STOP
    END IF
    i = i + 1
  END DO

END SUBROUTINE

