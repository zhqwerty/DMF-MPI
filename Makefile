./PHONY: main
main: mpi_run.cpp ./Example/examples.cpp
	mpic++ -std=c++11 -o mpi_run mpi_run.cpp ./Example/examples.cpp -larmadillo -lstdc++

./PHONY: single_thread
single_thread: single_thread.cpp ./Example/examples.cpp
	mpic++ -std=c++11 -o single_thread single_thread.cpp ./Example/examples.cpp -larmadillo

clean:
	@rm -f mpi_run single_thread
