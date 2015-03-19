# exactstrmatchgpu
Automatically exported from code.google.com/p/exactstrmatchgpu
Motivation

This was started because the author's intense interest at the awesome raw power of GPUs. As a novice GPU developer, the author does hope to become a very good one.
Where we are going with this

For starters, the author will concentrate on parallelizing exact string matching algorithms since this topic is the author's favourite for the next couple of months...
What's there now
29 Nov 2010

    Naive String Matcher
    Horspool String Matcher
    QuickSearch? String Matcher 

Serial Algorithm Characteristics
Naive String Matcher

    no preprocessing phase;
    constant extra space needed;
    always shifts the window by exactly 1 position to the right;
    comparisons can be done in any order;
    searching phase in O(mn) time complexity;
    2n expected text characters comparisons. 

Horspool String Matcher

    simplification of the Boyer-Moore algorithm;
    easy to implement;
    preprocessing phase in O(m+sigma) time and O(sigma) space complexity;
    searching phase in O(mn) time complexity;
    the average number of comparisons for one text character is between 1/sigma and 2/(sigma+1). 

QuickSearch? String Matcher

    simplification of the Boyer-Moore algorithm;
    uses only the bad-character shift;
    easy to implement;
    preprocessing phase in O(m+sigma) time and O(sigma) space complexity;
    searching phase in O(mn) time complexity;
    very fast in practice for short patterns and large alphabets. 

Future work

    Produce optimized versions of the algorithms (This should be done second)
    Produce more CUDA-ized algorithms (This should be done first) 

You can check out the Issues link on the recent progress of this project.

Meanwhile, feel free to use the code. Raise an issue if you see something wrong, or you like to raise a suggestion, improvements, comments etc. The author is not 100% aware of the current available literature w.r.t similar implementations, but he'll give proper references and attributions wherever applicable. 
