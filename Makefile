# test::test.cpp
#	g++ -o test test.cpp -larmadillo


main::main.cpp ./Example/examples.cpp
	mpicc -std=c++11 -o run main.cpp ./Example/examples.cpp -larmadillo

clean:
	-rm -f main
