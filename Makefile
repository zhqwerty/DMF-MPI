./PHONY: main
main: mpi_run.cpp ./Example/examples.cpp
	mpicc -std=c++11 -o mpi_run mpi_run.cpp ./Example/examples.cpp -larmadillo

./PHONY: single_thread
single_thread: single_thread.cpp ./Example/examples.cpp
	mpicc -std=c++11 -o single_thread single_thread.cpp ./Example/examples.cpp -larmadillo

#./PHONY: test
#test: ./Test/test.cpp
#	g++ -o test ./Test/test.cpp -larmadillo

clean:
	@rm -f mpi_run single_thread
