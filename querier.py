import json
import math
import re
import csv
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

# manual settings: adjust these as needed
corpus_file = "scifact/corpus.jsonl"      
query_file = "scifact/queries.jsonl"      
relevance_file = "scifact/qrels/test.tsv"   
results_file = "Results_title.txt"                
search_mode = "title"                        
max_results = 100                                

stopwords_text = """
a
about
above
ac
according
accordingly
across
actually
ad
adj
af
after
afterwards
again
against
al
albeit
all
almost
alone
along
already
als
also
although
always
am
among
amongst
an
and
another
any
anybody
anyhow
anyone
anything
anyway
anywhere
apart
apparently
are
aren
arise
around
as
aside
at
au
auf
aus
aux
av
avec
away
b
be
became
because
become
becomes
becoming
been
before
beforehand
began
begin
beginning
begins
behind
bei
being
below
beside
besides
best
better
between
beyond
billion
both
briefly
but
by
c
came
can
cannot
canst
caption
captions
certain
certainly
cf
choose
chooses
choosing
chose
chosen
clear
clearly
co
come
comes
con
contrariwise
cos
could
couldn
cu
d
da
dans
das
day
de
degli
dei
del
della
delle
dem
den
der
deren
des
di
did
didn
die
different
din
do
does
doesn
doing
don
done
dos
dost
double
down
du
dual
due
durch
during
e
each
ed
eg
eight
eighty
either
el
else
elsewhere
em
en
end
ended
ending
ends
enough
es
especially
et
etc
even
ever
every
everybody
everyone
everything
everywhere
except
excepts
excepted
excepting
exception
exclude
excluded
excludes
excluding
exclusive
f
fact
facts
far
farther
farthest
few
ff
fifty
finally
first
five
foer
follow
followed
follows
following
for
former
formerly
forth
forty
forward
found
four
fra
frequently
from
front
fuer
further
furthermore
furthest
g
gave
general
generally
get
gets
getting
give
given
gives
giving
go
going
gone
good
got
great
greater
h
had
haedly
half
halves
hardly
has
hasn
hast
hath
have
haven
having
he
hence
henceforth
her
here
hereabouts
hereafter
hereby
herein
hereto
hereupon
hers
herself
het
high
higher
highest
him
himself
hindmost
his
hither
how
however
howsoever
hundred
hundreds
i
ie
if
ihre
ii
im
immediately
important
in
inasmuch
inc
include
included
includes
including
indeed
indoors
inside
insomuch
instead
into
inward
is
isn
it
its
itself
j
ja
journal
journals
just
k
kai
keep
keeping
kept
kg
kind
kinds
km
l
la
large
largely
larger
largest
las
last
later
latter
latterly
le
least
les
less
lest
let
like
likely
little
ll
long
longer
los
low
lower
lowest
ltd
m
made
mainly
make
makes
making
many
may
maybe
me
meantime
meanwhile
med
might
million
mine
miss
mit
more
moreover
most
mostly
mr
mrs
ms
much
mug
must
my
myself
n
na
nach
namely
nas
near
nearly
necessarily
necessary
need
needs
needed
needing
neither
nel
nella
never
nevertheless
new
next
nine
ninety
no
nobody
none
nonetheless
noone
nope
nor
nos
not
note
noted
notes
noting
nothing
notwithstanding
now
nowadays
nowhere
o
obtain
obtained
obtaining
obtains
och
of
off
often
og
ohne
ok
old
om
on
once
onceone
one
only
onto
or
ot
other
others
otherwise
ou
ought
our
ours
ourselves
out
outside
over
overall
owing
own
p
par
para
particular
particularly
past
per
perhaps
please
plenty
plus
por
possible
possibly
pour
poured
pouring
pours
predominantly
previously
pro
probably
prompt
promptly
provide
provides
provided
providing
q
quite
r
rather
re
ready
really
recent
recently
regardless
relatively
respectively
round
s
said
same
sang
save
saw
say
second
see
seeing
seem
seemed
seeming
seems
seen
sees
seldom
self
selves
send
sending
sends
sent
ses
seven
seventy
several
shall
shalt
she
short
should
shouldn
show
showed
showing
shown
shows
si
sideways
significant
similar
similarly
simple
simply
since
sing
single
six
sixty
sleep
sleeping
sleeps
slept
slew
slightly
small
smote
so
sobre
some
somebody
somehow
someone
something
sometime
sometimes
somewhat
somewhere
soon
spake
spat
speek
speeks
spit
spits
spitting
spoke
spoken
sprang
sprung
staves
still
stop
strongly
substantially
successfully
such
sui
sulla
sung
supposing
sur
t
take
taken
takes
taking
te
ten
tes
than
that
the
thee
their
theirs
them
themselves
then
thence
thenceforth
there
thereabout
thereabouts
thereafter
thereby
therefor
therefore
therein
thereof
thereon
thereto
thereupon
these
they
thing
things
third
thirty
this
those
thou
though
thousand
thousands
three
thrice
through
throughout
thru
thus
thy
thyself
til
till
time
times
tis
to
together
too
tot
tou
toward
towards
trillion
trillions
twenty
two
u
ueber
ugh
uit
un
unable
und
under
underneath
unless
unlike
unlikely
until
up
upon
upward
us
use
used
useful
usefully
user
users
uses
using
usually
v
van
various
ve
very
via
vom
von
voor
vs
w
want
was
wasn
way
ways
we
week
weeks
well
went
were
weren
what
whatever
whatsoever
when
whence
whenever
whensoever
where
whereabouts
whereafter
whereas
whereat
whereby
wherefore
wherefrom
wherein
whereinto
whereof
whereon
wheresoever
whereto
whereunto
whereupon
wherever
wherewith
whether
whew
which
whichever
whichsoever
while
whilst
whither
who
whoever
whole
whom
whomever
whomsoever
whose
whosoever
why
wide
widely
will
wilt
with
within
without
won
worse
worst
would
wouldn
wow
x
xauthor
xcal
xnote
xother
xsubj
y
ye
year
yes
yet
yipee
you
your
yours
yourself
yourselves
yu
z
za
ze
zu
zum
""".strip().split()

stopwords_set = set(stopwords_text)

# make everything lowercase, remove stopwords, stem
def preproc(txt):
    stemmer = PorterStemmer()
    txt = txt.lower()
    tokens = re.split(r"[^a-zA-Z]+", txt)
    tokens = [w for w in tokens if w]
    tokens = [w for w in tokens if w not in stopwords_set]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens

def load_corpus(fp, mode="full"):
    docs = {}
    titles = {}
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            did = str(data["_id"])
            ttl = data.get("title", "")
            txt = data.get("text", "")

            titles[did] = ttl
            if mode == "title":
                combo = ttl
            else:
                combo = ttl + " " + txt
            docs[did] = combo
    return docs, titles

# step 2: indexing ndex: term -> {doc_id -> frequency}, doc_token_counts: doc_id -> total token count
def build_inverted_index(documents):
    index = defaultdict(lambda: defaultdict(int))
    doc_token_counts = defaultdict(int)

    for doc_id, text in documents.items():
        tokens = preproc(text)
        doc_token_counts[doc_id] = len(tokens)
        freqs = Counter(tokens)
        for term, count in freqs.items():
            index[term][doc_id] += count

    return index, doc_token_counts

#    idf = log10( (num_docs - df + 0.5)/ (df + 0.5) )
def compute_idf(index, num_docs):
    idf_scores = {}
    for term, posting in index.items():
        doc_freq = len(posting)
        idf_scores[term] = math.log10((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    return idf_scores

# doc_vectors[doc_id][term] = bm25-like tf * idf
#doc_lengths[doc_id] = sqrt of sum-of-squares
def build_doc_vectors(index, idf_scores):
    doc_vectors = defaultdict(dict)
    doc_lengths = defaultdict(float)
    doc_sizes = defaultdict(int)
    for term, posting in index.items():
        for doc_id, freq in posting.items():
            doc_sizes[doc_id] += freq

    total_size = sum(doc_sizes.values())
    avg_doc_size = total_size / len(doc_sizes) if doc_sizes else 1.0

    k1 = 1.5  
    b = 0.75  

    for term, posting in index.items():
        term_weight = idf_scores[term]
        for doc_id, term_freq in posting.items():
            # bm25
            doc_size = doc_sizes[doc_id]
            norm_tf = ((k1 + 1) * term_freq) / (k1 * ((1 - b) + b * doc_size /   avg_doc_size) + term_freq)
            final_weight = norm_tf * term_weight
            doc_vectors[doc_id][term] = final_weight
    for doc_id, weights in doc_vectors.items():
        length_squared = sum(w * w for w in weights.values())
        doc_lengths[doc_id] = math.sqrt(length_squared)

    return doc_vectors, doc_lengths

# step 3: retrieval
# compute tf-idf weights for query terms with improved weighting
def compute_query_vector(query_text, idf_scores):
    tokens = preproc(query_text)
    term_freqs = Counter(tokens)
    query_weights = {}
    k1 = 1.5  
    for term, freq in term_freqs.items():
        if term in idf_scores:
        
            norm_tf = ((k1 + 1) * freq) / (k1 + freq)
            query_weights[term] = norm_tf * idf_scores[term]
        else:
            query_weights[term] = 0.0
    return query_weights

def cosine_similarity(query_weights, doc_weights, doc_length):
    dot_product = 0.0
    for term, query_weight in query_weights.items():
        doc_weight = doc_weights.get(term, 0.0)
        dot_product += query_weight * doc_weight

    query_length_sq = sum(w * w for w in query_weights.values())
    query_length = math.sqrt(query_length_sq)

    denominator = query_length * doc_length
    if denominator == 0.0:
        return 0.0
    return dot_product / denominator

def expand_query_with_feedback(query_text, relevant_docs, documents, idf_scores, param=1.0, param2=0.75):
    query_tokens = preproc(query_text)
    query_freqs = Counter(query_tokens)

    relevant_vectors = []
    for doc_id in relevant_docs:
        if doc_id in documents:
            doc_tokens = preproc(documents[doc_id])
            relevant_vectors.append(Counter(doc_tokens))

    if not relevant_vectors:
        return query_text

    expanded_freqs = Counter()
    for term, freq in query_freqs.items():
        expanded_freqs[term] = param * freq

    num_relevant = len(relevant_vectors)
    for vector in relevant_vectors:
        for term, freq in vector.items():
            expanded_freqs[term] += (param2 * freq) / num_relevant

    top_terms = sorted(expanded_freqs.items(), key=lambda x: x[1], reverse=True)[:20]
    expanded_query = query_text + " " + " ".join(term for term, _ in top_terms if term not in query_tokens)
    return expanded_query

 #score docs by cosine similarity, optional pseudo-relevance feedback
def retrieve(query_text, doc_vectors, doc_lengths, idf_scores, qrels=None, qid=None, documents=None, top_k=100):
   
    if qrels and qid and documents and qid in qrels:
        relevant_docs = [doc_id for doc_id, score in qrels[qid].items() if score == '1']
        if relevant_docs:
            query_text = expand_query_with_feedback(query_text, relevant_docs, documents, idf_scores)
    
    query_weights = compute_query_vector(query_text, idf_scores)
    search_results = []
    
    for doc_id, doc_weights in doc_vectors.items():
        similarity = cosine_similarity(query_weights, doc_weights, doc_lengths[doc_id])
        search_results.append((doc_id, similarity))

    search_results.sort(key=lambda x: x[1], reverse=True)
    return search_results[:top_k]

def load_queries(query_path):
    """
    load queries from jsonl file
    """
    query_dict = {}
    with open(query_path, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            query_id = str(data["_id"])
            query_dict[query_id] = data["text"]
    return query_dict

def load_qrels(relevance_path):
    test_query_ids = set()
    relevance_data = defaultdict(dict)
    with open(relevance_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            query_id, doc_id, relevance = row
            test_query_ids.add(query_id)
            relevance_data[query_id][doc_id] = relevance
    return test_query_ids, relevance_data

def main():
    print(f"loading corpus from {corpus_file} (index_mode={search_mode})")
    documents, titles = load_corpus(corpus_file, search_mode)
    print(f" => {len(documents)} documents loaded.")

    print("building inverted index ...")
    index, doc_token_counts = build_inverted_index(documents)
    num_docs = len(documents)

    idf_scores = compute_idf(index, num_docs)
    doc_vectors, doc_lengths = build_doc_vectors(index, idf_scores)
    print(f"vocabulary size: {len(index)}")

    print("\nvocabulary sample (first 100 terms):")
    sample_terms = sorted(list(index.keys()))[:100]
    for i, term in enumerate(sample_terms):
        if i and i % 10 == 0:
            print()
        print(f"{term}, ", end="")
    print("\n")

    print(f"loading queries from {query_file}")
    query_map = load_queries(query_file)

    print(f"loading test set from {relevance_file}")
    test_queries, relevance_data = load_qrels(relevance_file)

    query_list = []
    sorted_queries = sorted(list(test_queries), key=lambda x: int(x))
    for query_id in sorted_queries:
        if int(query_id) % 2 == 1:
            query_list.append(query_id)

    print(f"total queries used (odd ids only): {len(query_list)}")
    print("\nfirst few test queries:")
    for query_id in query_list[:5]:
        print(f"query {query_id}: {query_map[query_id]}")
    print()

    print(f"retrieving top {max_results} docs per query ...")
    run_id = "run_vsm_full" if search_mode == "full" else "run_vsm_title"
    output_path = results_file

    queries_shown = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for query_id in query_list:
            query_text = query_map[query_id]
            search_results = retrieve(query_text, doc_vectors, doc_lengths, idf_scores,
                               qrels=relevance_data, qid=query_id, documents=documents,
                               top_k=max_results)

            if queries_shown < 2:
                print(f"\ntop 10 for query {query_id}: {query_text}")
                print("-" * 80)
                for i, (doc_id, score) in enumerate(search_results[:10]):
                    print(f"{i+1}. (score: {score:.4f}) {titles[doc_id]}")
                print("-" * 80)
                queries_shown += 1

            rank = 1
            for (doc_id, score) in search_results:
                outfile.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_id}\n")
                rank += 1

    print(f"done. wrote output to {output_path}.")
    print("evaluate with trec_eval, e.g.:")
    print(f"  trec_eval {relevance_file} {output_path}")

if __name__ == "__main__":
    main()