# test::test.cpp
#	g++ -o test test.cpp -larmadillo


main::mpi_run.cpp ./Example/examples.cpp
	mpicc -std=c++11 -o mpi_run mpi_run.cpp ./Example/examples.cpp -larmadillo

#main::single_thread.cpp ./Example/examples.cpp
#	mpicc -std=c++11 -o single_thread single_thread.cpp ./Example/examples.cpp -larmadillo

clean:
	-rm -f mpirun
