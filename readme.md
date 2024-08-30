This the aim of this project is to extract and analyze some quick metrics from data leaks from different services.

For ethical reasons the data leaks or sources will not be uploaded.

Everything runs on virtual enviroment

Instructions to get the extraction and analysis of the data from each leak:

1. Download leak data
2. Place the txt in format user:pass in the folder with the name of the leak and inside folder data
    In the form "./leakname/data"
3. Populate leaks.txt with each leakname in a newline
4. Execute master.py

Stats:
    Mask: password mask format in luds format (l for lower, u for upper, d for decimal, s for special)
    Length
    simpleentropy: E = L × log(R) / log(2)
    shannonentropy: H(X) = - Σ [p(x) * log2(p(x))] 
    Is common if the password is in the list of the 100 most common passwords
    Score in zxcvbn
    Guesses in zxcvbn

Extraction modes:
    Some specific data bases have special formats for users and passwords, for example some of them have some passwords coded as $HEX["ascii code"], the new mode code has to be added to dataextract and the name added to master. The mode is then selected adding the name given to the mode after the leakname in leaks.txt (such as sharethis $HEX) 

Single stats:
    Sometimes it is not needed to use all stats, instead adding the name of the stat following the leak name calculates the single stat needed. The current implementation works for simple_entropy, shannon_entropy and password strength.
