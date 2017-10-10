# test::test.cpp
#	g++ -o test test.cpp -larmadillo


main::main.cpp examples.cpp
	mpicc -std=c++11 -o main main.cpp examples.cpp -larmadillo

clean:
	-rm -f main
