<h1>Project Overview</h1>
<p>The aim of this project is to extract and analyze metrics from data leaks across various services. For ethical reasons, data leaks or sources will not be uploaded.</p>

<h1>Environment Setup</h1>
<p>Everything runs in a virtual environment.</p>

<h1>Instructions for Data Extraction and Analysis</h1>
<ol>
    <li>Download the leak data.</li>
    <li>Place the <code>.txt</code> file in the format <code>user:pass</code> inside a folder named after the leak, in the <code>data</code> subfolder (e.g., <code>./leakname/data</code>).</li>
    <li>Populate <code>leaks.txt</code> or <code>leak_types.txt</code> with each leak name on a new line.</li>
    <li>Execute <code>master.py</code> with the desired option: <code>py master.py [option]</code>.</li>
</ol>

<h2>Available Options:</h2>
<ul>
    <li><code>-s --stats</code>:
        Process leaks and gather statistics (from <code>leaks.txt</code>).
    </li>
    <li><code>-d --distributioncomparison</code>:
        Get distribution comparison (from <code>leak_types.txt</code>).<br/>
        <span style="margin-left: 20px;">(Note: Ensure <code>stats.txt</code> and <code>password_score_and_length.txt</code> files are generated or in the correct location.)</span>
    </li>
    <li><code>-l --latex</code>:
        Generate a LaTeX file in the latex folder with important data (from <code>leak_types.txt</code>).
    </li>
    <li><code>-c --cluster</code>:
        Execute the clustering module. For K-means clustering, append <code>kmeans</code> after the option.
    </li>
    <li><code>-sa --serviceanalisis</code>:
        Conduct analysis on service-specific data from leaks.
    </li>
    <li><code>-lr --leakregression</code>:
        Perform regression analysis between password strength and risk values in leaks.
    </li>
    <li><code>-h --help</code>:
        Display help menu.
    </li>
</ul>


<h1>Format for <code>leaks.txt</code></h1>
<p>Each line should be in the format: <code>&lt;leakname&gt; &lt;stat&gt; &lt;extraction mode&gt;</code>. Lines starting with <code>#</code> are ignored. Example:</p>
<pre>
linkedin leak
#mate1
myheritage simple_entropy
sharethis password_score_and_length extract_hex
#shein
</pre>

<h1>Distribution Comparison</h1>
<p>This generates graphs in the <code>figures</code> folder to analyze similarities between data leaks. It creates Kullback-Leibler matrices for masks by length, scores by length, and global scores, along with histograms for scores and scores by length.</p>

<h1>Metrics Description</h1>
<ul>
    <li><strong>Mask</strong>: Password mask format in <code>luds</code> (l = lower, u = upper, d = decimal, s = special).</li>
    <li><strong>Length</strong>: Length of the password.</li>
    <li><strong>Simple Entropy</strong>: <code>E = L × log(R) / log(2)</code>.</li>
    <li><strong>Shannon Entropy</strong>: <code>H(X) = - Σ [p(x) * log2(p(x))]</code>.</li>
    <li><strong>Commonness</strong>: Indicates if the password is among the 100 most common.</li>
    <li><strong>Score</strong>: Zxcvbn score.</li>
    <li><strong>Guesses</strong>: Zxcvbn guess estimates.</li>
</ul>

<h1>Extraction Modes</h1>
<p>Specific databases may have unique formats for usernames and passwords (e.g., passwords coded as <code>$HEX["ascii code"]</code>). New modes must be added to <code>dataextract</code> and referenced in <code>master.py</code>. Specify the mode in <code>leaks.txt</code> as follows: <code>sharethis extract_hex</code>.</p>

<h1>Single Stats</h1>
<p>If only a specific stat is needed, append the stat name after the leak name in <code>leaks.txt</code>. Supported stats include: <code>simple_entropy</code>, <code>shannon_entropy</code>, <code>password_strength</code>, and <code>password_score_and_length</code>.</p>
