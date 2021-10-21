import os



def Poisson_equation_2D(Lx,Lz,Rhs,Const):
    Nx,Nz=size(Rhs);

    #Specifying parameters (check why It does not work if we change them)

    dx=Lx/(Nx-1);                      # Width of space step(x)
    dy=Lz/(Nz-1);                      # Width of space step(y)
    x=0:dx:Lx;                         # Range of x(0,2) and specifying the grid points
    y=0:dy:Lz;                         # Range of y(0,2) and specifying the grid points
    b=zeros(Nx,Nz);                    # Preallocating b
    pn=zeros(Nx,Nz);                   # Preallocating pn


    % Initial Conditions
    p=zeros(Nx,Nz);                    # Preallocating p
s

    Rhs=Const.*fliplr(Rhs);   
    b=(Rhs);

    
    i=2:Nx-1;
    j=2:Nz-1;

    #Poisson equation solution (iterative) method

    tol = 1e-4;			 % Set tollerance
    maxerr = inf;	     % initial error
    iter = 0;
    pn=p;

    while maxerr > tol
        iter = iter + 1;
        #disp(['Iteration no. ',num2str(iter)]);

        
        #Explicit iterative scheme with C.D in space (5-point difference)

        p(i,j)=((dy^2*(pn(i+1,j)+pn(i-1,j)))+(dx^2*(pn(i,j+1)+pn(i,j-1)))-(b(i,j)*dx^2*dy*2))/(2*(dx^2+dy^2));
        
        #Boundary conditions     
        #Neumann's conditions   % dp/dx|end=dp/dx|end-1 
        p(1,:)=p(2,:);
        p(end,:)=p(end-1,:);
        
        
    #     Dirichlet's conditions 
    #     p(:,1)=1.43;
    #     p(:,end)=1.32;
        
    ## Neumann's conditions
        p(:,1)=p(:,2);
        p(:,end)=p(:,end-1);

        maxerr = max(max(abs((p-pn)./p)));
        #disp(['Maximum error is  ',num2str(maxerr)]);
        pn=p;
    
	#as long the error larger than tolerance, continue


PG2_gray=p*255;
n_max=1.43;    
n_min=1.332;
n2= scaledata(PG2_gray,n_min,n_max);
    