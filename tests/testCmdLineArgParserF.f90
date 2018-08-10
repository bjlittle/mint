program test
    ! defines the C API
    use mnt_cmdlineargparser_capi_mod
    use, intrinsic :: iso_c_binding, only: c_loc

    use, intrinsic :: iso_c_binding, only: c_size_t, c_int, c_double

    implicit none
    integer(c_size_t)                                :: prsr
    integer(c_int)                                   :: ier, nargs, i, n
    character(len=mnt_string_size), allocatable      :: args(:)
    character(len=mnt_string_size)                   :: argv_full
    integer(c_int)                                   :: ival
    real(c_double)                                   :: dval
    character(len=mnt_string_size)                   :: sval, s2, s3

    ier = mnt_cmdlineargparser_new(prsr)

    ! set
    call mnt_make_c_string("some integer", s3)
    ier = mnt_cmdlineargparser_setint(prsr, "-i"//char(0), -123, s3)

    call mnt_make_c_string("some double", s3)
    ier = mnt_cmdlineargparser_setdouble(prsr, "-d"//char(0), -1.23_c_double, s3)

    call mnt_make_c_string("Hello there!", s2)
    call mnt_make_c_string("some integer", s3)
    ier = mnt_cmdlineargparser_setstring(prsr, "-s"//char(0), s2, s3)

    ! parse
    nargs = command_argument_count() + 1
    allocate(args(nargs))
    do i = 1, nargs
        call get_command_argument(i, argv_full)
        ! add termination character, trim...
        call mnt_make_c_string(argv_full, args(i))
    enddo
    n = len(args(1))
    ier = mnt_cmdlineargparser_parse(prsr, nargs, n, args)

    ! extract
    ier = mnt_cmdlineargparser_getint(prsr, "-i"//char(0), ival)
    ier = mnt_cmdlineargparser_getdouble(prsr, "-d"//char(0), dval)
    sval(:) = ' '
    ier = mnt_cmdlineargparser_getstring(prsr, "-s"//char(0), sval, n)

    print *, "-i arg is ", ival
    print *, "-d arg is ", dval
    print *, "-s arg is ", trim(sval)

    ! clean up
    deallocate(args)
    ier = mnt_cmdlineargparser_del(prsr)

end program 
