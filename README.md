# README #


### What is this repository for? ###

* parallelGram is a simple program for counting Google N-grams word co-occurrences in parallel. It offers three search mechanisms: pattern matching, exact matching, and exact matching with negation exclusion. This is an early version and there are plenty of inefficiencies. Nonetheless, the parallel computing greatly speeds up the search. 
* Version: 0.1

### How do I get set up? ###

The current version requires that you have the Google N-grams data that you wish to search stored locally. To download ngrams, I recommend using the Google-Ngram-Downloader python package. To install this package, you can use pip: 

`pip install google-ngram-downloader`

You can then run google-ngram-downloader's command line tool from you terminal. To download 5 grams, you could use a call like: 

`google-ngram-downloader download -n 5 -v -l eng`

This will download english (-l eng) 5 (-n 5) grams, print current opperations in terminal (-v), and save them in a nested directory that will be created in your current working directory. 

* parallelGram was written for Python 2.7. 

* You need these dependencies: 

numpy as np
pandas as pd
pickle
gzip
io
requests
urllib
dill
multiprocessing`

If you are using anaconda, many of these will already be installed. Regardless, you can stall all of these packages with pip.

* **Setupd** parallelGram is designed to be run using a single command line call that directs the program to a simple configuration file. This configuration file needs to have at least 6 components, one on each line. You can create a configuration file as a simple, non-formatted text file. See below for example:

### Configuration.txt contents

Line 1:  full path to the directory containing the ngram files<br>

Line 2:  Number of jobs to split this across. This number could be a function of the number of cores your computer has, but also how intensively you'll be using your computer while this is running. <br>

Line 3: The directory where you want the output to be stored. You'll get 2 kinds of output, python formatted files that will be spit out by each of the jobs and two (one contains counting the frequencies and the other containing number of volumes the n-gram or n-gram pair occurred in) csv files containing the co-occurrences. <br>

line 4: full path to a .txt file containing the base words that you want to search for. These words should all be on one line (no line breaks) and they should be separated by a comma or a comma + space<br>

line 5: This is the path to the target words .txt file. It is the same as the base words file, but specifies the other set of words you want to look for.<br> 

line 6: This line should contain a 1, 2, or 3, which specifies which search procedure to use. 1 conducts pattern matching Thus, if you're looking for cad in a phrase that contains abracadabra, search 1 will count 'cad' in abracadabra as a hit (this is the fastest approach). 2 conducts exact matching, thus only 'cad' will be counted as a hit for 'cad'. 3 conducts exact matching, too, but it skips matches that are negated. Thus, 'he is not a cad' would *not* be counted as a hit *if* you specify 'not a' as a negation word or phrase. <br>

line 7: If you use exact matching with negation, you need to also provide a path to a file containing negation words/phrases. This words/phrases should be comma separated and all on one line. <br>



### Example of configuration.txt contents (don't use blank lines): 


/path/to/ngram/directory

3

/output/directory

/path/to/basewords.txt

/path/to/targetwords.txt

3

/path/to/negationWords.txt



Once you have the configuration file and all of the other requisite files setup, you can run parallelGram using a command like: 

`python parallelGram /path/to/configuration/file` 

### Contribution guidelines ###

* Feel free to make this work better. 

### Author? ###

* Joe Hoover
* jehoover@usc.edu