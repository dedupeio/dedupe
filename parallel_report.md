# Benchmarking parallel scoreDuplicates:

I'm using python's Pool.map() to make this proof of concept. No fancy map-reduce stuff in here.

The calling of the core.scoreDuplicates function has been parallelized. Clustering and blocking is still as usual. Blocking currently seems to be a huge bottleneck (if the file size grows over 4-5 times bigger, it keeps running for over 4 mins; didn't check till completion) than the original csv test dataset, will try adding parallelism to it too.

examples/csv_example/duplicate.py has been added - it basically makes the test dataset 4 times bigger by duplicating it, taking care of the repeating sequential IDs.

Here is a sample run on my Intel i5-2500K quad-core desktop with 4GB RAM, running Ubuntu 11.10, 32 bit. 8 processes are run by pool. This was done on the bigger dataset, named custom_big.csv, generated with duplicate.py.

<pre>
$ time python ex*/csv*/duplicate.py

real	0m0.627s
user	0m0.612s
sys	0m0.012s

$ time python examples/csv_example/csv_example.py
importing data ...
reading from csv_example_learned_settings
blocking...
clustering in serial mode...
serial scoreDuplicates takes :  70.3518688679
clustering in parallel mode...
Parallel scoreDuplicates with  8  processes takes :  25.2054569721
# duplicate sets 825

real	2m54.906s
user	3m55.975s
sys	0m1.716s

</pre>

Sample run on the same environment with the original csv dataset:
<pre>
$ python examp*/csv*/csv_example.py
importing data ...
reading from csv_example_learned_settings
blocking...
clustering in serial mode...
Serial scoreDuplicates takes :  6.34111309052
clustering in parallel mode...
Parallel scoreDuplicates with  8  processes takes :  2.5353000164
# duplicate sets 527
</pre>
