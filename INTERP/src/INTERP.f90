SUBROUTINE INTERP(Nr, Nr2, qdegree, xcor, xcor2, DD0)
   USE :: Piecewise_Polynomial_Interpolation

   IMPLICIT NONE
   integer, intent(in) :: Nr, Nr2, qdegree
   real, intent(out) :: DD0(Nr2, Nr)
   real, intent(in) :: xcor(Nr), xcor2(Nr2)

   integer j,i,jcor,icor,ierr,derv(1)
   real d(Nr2,1,Nr)
   real res, xLr, deltaxcor, l, s

   ! ... Check inputs ...
   derv(1) = 0;
   IF (Nr .LT. qdegree) STOP 'N too small for MH scheme'
  
    
   Polynomial_Degree=qdegree
   call Interpolation_Matrix(Grid_Nodes        = xcor,  &
                          Interpolation_Points = xcor2,  &
                          Derivatives          = derv,  &
                          Matrix               = d)

   
   DD0 = d(:, 1, :)
end subroutine INTERP