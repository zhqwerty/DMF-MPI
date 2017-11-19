# test::test.cpp
#	g++ -o test test.cpp -larmadillo


main::new_main.cpp ./Example/examples.cpp
	mpicc -std=c++11 -o run new_main.cpp ./Example/examples.cpp -larmadillo

#main::single_thread.cpp ./Example/examples.cpp
#	mpicc -std=c++11 -o single_thread single_thread.cpp ./Example/examples.cpp -larmadillo

clean:
	-rm -f main
