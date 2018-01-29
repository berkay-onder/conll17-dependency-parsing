# Infinite stream processing version 2, don't give <unk> as an input to model, which can handle infinite amount of tokenized-raw text data

const PosTag = UInt8         # 17 universal part-of-speech tags
# typealias Word Tuple{UInt16,String}
# typealias Arc Tuple{Int,Int}
# typealias Arcset Array{Arc,1}

const SOS = "<s>"
const EOS = "</s>"
const UNK = "<unk>"

# Universal POS tags (17)
const UPOSTAG = Dict{String,PosTag}(
"ADJ"   => 1, # adjective
"ADP"   => 2, # adposition
"ADV"   => 3, # adverb
"AUX"   => 4, # auxiliary
"CCONJ" => 5, # coordinating conjunction
"DET"   => 6, # determiner
"INTJ"  => 7, # interjection
"NOUN"  => 8, # noun
"NUM"   => 9, # numeral
"PART"  => 10, # particle
"PRON"  => 11, # pronoun
"PROPN" => 12, # proper noun
"PUNCT" => 13, # punctuation
"SCONJ" => 14, # subordinating conjunction
"SYM"   => 15, # symbol
"VERB"  => 16, # verb
"X"     => 17, # other
)

function create_vocab(vocabfile::AbstractString)
    ercount = 0
    result = Dict{AbstractString, Int}(SOS=>1, EOS=>2, UNK=>3)
    open(vocabfile) do f
        for line in eachline(f)
            words= split(line)
            try
                @assert(length(words) == 1, "The vocabulary file seems broken")
            catch e
                ercount += 1
                if (length(words) == 0)
                    if (ercount == 5)
                        info("There is empty line in vocabulary file")
                        ercount = 0
                    end
                    continue
                else
                    warn("Something unexpected happen in vocabfile")
                end
            end
            word = words[1]
            if length(word) > 65
                #isinteractive() && warn("Too long words in file $vocabfile")
                word = word[1:65]
            end
            get!(result, word, 1+length(result))
        end
    end
    return result
end

function read_conllfile(conllfile)
    sentences=[]
    postaglists=[]
    parentlists=[]
    conll_input=open(conllfile)
    while !eof(conll_input)
        input_line=readline(conll_input)
        if startswith(input_line,"# text =")
            input_line=readline(conll_input)
            new_sentence=[]
            new_postaglist=[]
            new_parentlist=[]
            while length(input_line)>0
                conllfields=split(input_line,"\t")
                push!(new_sentence,String(conllfields[2]))
                push!(new_postaglist,String(conllfields[4]))
                push!(new_parentlist,parse(conllfields[7]))
                input_line=readline(conll_input)
            end
            push!(sentences,new_sentence)
	    push!(postaglists,new_postaglist)
            push!(parentlists,new_parentlist)
        end
    end
    close(conll_input)
    sentences,postaglists,parentlists
end

function Chu_Liu_Edmonds(V,lambda)
    n=length(V)
end

function CKY(V,lambda)
    n=length(V)
    C=zeros(n+1,n+1,n+1)
    A=fill(Set(),(n+1,n+1,n+1))
    for l=1:n #subsentence length
        for s=1:n+1
            t=s+l
            if t<=n+1
                for i=s:t
                    max_score=max_q=max_j=0
                    for q=s:t-1
                        for j=s:t
                            if j>i && s<=i<=q && q<j<=t
                                tree_score=C[s,q,i]+C[q+1,t,j]+lambda[i,j]
                                if tree_score>max_score
                                    max_score=tree_score
                                    max_q=q
                                    max_j=j
                                end
                            end
                            if 1<j<i && s<=j<=q && q<i<=t
                                tree_score=C[s,q,j]+C[q+1,t,i]+lambda[i,j]
                                if tree_score>max_score
                                    max_score=tree_score
                                    max_q=q
                                    max_j=j
                                end
                            end
                        end
                    end
                    if C[s,t,i]<max_score
                        C[s,t,i]=max_score
                        if max_j>i && s<=i<=max_q && max_q<max_j<=t
                            A[s,t,i]=union(A[s,max_q,i],A[max_q+1,t,max_j],Set([(i-1,max_j-1)]))
                        end
                        if 1<max_j<i && s<=max_j<=max_q && max_q<i<=t
                            A[s,t,i]=union(A[s,max_q,max_j],A[max_q+1,t,i],Set([(i-1,max_j-1)]))
                        end
                    end
                end
            end
        end
    end
    return C[1,n+1,1],A[1,n+1,1]
end

result=create_vocab("wordcount.txt")
#loadcorpus("en-ud-dev.conllu",result)
fsize=length(result) # size of feature vector
#w=randn(4*fsize+68)
#fvector=Array{Float64}(4*fsize+68)
sentences,postaglists,parentlists=read_conllfile("en-ud-dev.conllu")
for i=1:length(sentences)
    n=length(sentences[i])
    lambda=zeros(n+1,n+1)
    for j=1:length(parentlists[i])
        lambda[parentlists[i][j]+1,j+1]=1
    end
    display(CKY(sentences[i],lambda))
end
#CKY(sentences[1])
#println(length(sentences))
#for V in sentences
#    n=length(V)
#    A=[(0,1)]  # Arcset
#    for i=2:n
#        push!(A,(0,i))
#    end
#    for i=1:n
#        for j=1:n
#            if i!=j
#                push!(A,(i,j))
#            end
#        end
#    end
#    G=(V,A)
#    lambda=rand(n+1,n)
#    for i=1:n
#        lambda[i+1,i]=0
#    end
#    Chu_Liu_Edmonds(G,lambda)
#end
