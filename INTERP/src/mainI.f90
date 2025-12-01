program mainI
  implicit none
  integer :: i
  integer :: Nr, Nr2, q
  real, allocatable, dimension(:, :) :: DD0
  real, allocatable, dimension(:) :: xcor, xcor2

  open(1,file='../INTERP/inputs/INTERP.dat')
     read(1,*) Nr,Nr2,q
  allocate(xcor(Nr), xcor2(Nr2))

     do i = 1, Nr
      read(1, *) xcor(i)
     enddo

    do i = 1, Nr2
      read(1, *) xcor2(i)
    enddo
  close(1)

  allocate(DD0(Nr2, Nr))

  call INTERP(Nr, Nr2, q, xcor, xcor2, DD0)
  
  open(unit = 1, file = '../INTERP/output/DD0.dat')
  do i = 1, Nr2
    write(1, *) DD0(i, :)
  enddo
  close(1)

deallocate(DD0,xcor,xcor2)

end program mainI
