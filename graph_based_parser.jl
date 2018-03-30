using Knet

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

#=
function Chu_Liu_Edmonds(V,lambda)
    A=[]
    n=length(V)
    for i=1:n
        max_value=maximum(lambda[i,:])
        max_index=findfirst(a->a==max_value, lambda[i,:])
        push!(A,(max_index,i))
    end
    A
    
end
=#

# minimum spanning tree algoritmaları incelenecek

# parsing paper'larında algoritmalar ve loss function'lar incelenecek

function Chu_Liu_Edmonds(V,lambda)
    A_prime=[]
    n=length(V)
    for i=1:n+1
        max_score=maximum(lambda[:,i])
        if max_score>0
            max_index=findfirst( score -> score == max_score, lambda[:,i] )
            push!(A_prime,(max_index-1,i-1))
        end
    end
    visited=[]
    cycle=nothing
    for pair in A_prime
        head=pair[1]
        push!(visited,pair)
#        parents=filter(pair -> pair[2] == head, A_prime)
        slowindex=findfirst( arc -> arc[2] == head, A_prime )
        while slowindex>0
            slow=A_prime[slowindex]
            cycle_start=findfirst( arc -> arc == slow, visited )
            if cycle_start>0
                cycle=visited[cycle_start:end]
            end
            head=slow[1]
            push!(visited,slow)
            slowindex=findfirst( arc -> arc[2] == head, A_prime )
        end
    end
    cycle,A_prime
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

# Eisner seems to be working now :)))))
function Eisner(S, lambda)
    n=length(S)
    E=Array{Any}(n,n,2,2)
    #=for s=1:n
        for d=1:2 c=1:2
            E[s,s,d,c]=0
        end
    end=#
    E=Array{Any}(zeros(n+1,n+1,2,2))
    A=fill(Set(),(n+1,n+1,2,2))
    #A=Array{Any}(n,n,2,2)
    #temp_scores=zeros(2,2)
    #temp_sets=fill(Set(),(2,2))
    for m=1:n+1
        for s=1:n+1
            t=s+m
            if t>n+1 break
            end
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,2,1]+E[q+1,t,1,1]+lambda[t,s]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[1,2]=max_score
            E[s,t,1,2]=max_score
            if max_q>0
                #temp_sets[1,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(t,s)]))
                A[s,t,1,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(t-1,s-1)]))
            #else 
                #A[s,t,1,2]=union(A[s,s,2,1],A[s+1,t,1,1])
            end
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,2,1]+E[q+1,t,1,1]+lambda[s,t]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[2,2]=max_score
            E[s,t,2,2]=max_score
            if max_q>0
                #temp_sets[2,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(s,t)]))
                A[s,t,2,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(s-1,t-1)]))
            #else
                #A[s,t,2,2]=union(A[s,s,2,1],A[s+1,t,1,1])
            end
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,1,1]+E[q,t,1,2]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[1,1]=max_score
            E[s,t,1,1]=max_score
            if max_q>0
                #temp_sets[1,1]=union(A[s,max_q,1,1],A[max_q,t,1,2])
                A[s,t,1,1]=union(A[s,max_q,1,1],A[max_q,t,1,2])
            end
            max_score=max_q=-1
            
            for q=s+1:t
                tree_score=E[s,q,2,2]+E[q,t,2,1]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[2,1]=max_score
            E[s,t,2,1]=max_score
            if max_q>0
                #temp_sets[2,1]=union(A[s,max_q,2,2],A[max_q,t,2,1])
                A[s,t,2,1]=union(A[s,max_q,2,2],A[max_q,t,2,1])
            end
            #=for i=1:2 j=1:2
               E[s,t,i,j]=temp_scores[i,j]
               A[s,t,i,j]=temp_sets[i,j]
            end=#
        end
    end
    #=dependents=ones(n)
    for pair in A[1,n,2,1]
        dependents[pair[2]]=0
    end
    push!(A[1,n,2,1],(0,findfirst(dependents)))=#
    E[1,n+1,2,1],A[1,n+1,2,1]
end
# I gonna cry now :'|

#sigm(x) = (1./(1+exp(-x)))

# logistic loss, sigmoid
# negative log probability uygulanacak doğru cevaplara
# sum p log_phat + (1-p) log (1-phat)
#knet Array'erde loop yerine vectorize işlemler yapılacak
#elementwise sigmoid uygulanacak

function mse(ygold, ypred)
    sum((ygold-ypred).^2)
end

function mlp(w, x, ygold)
    y1 = sigm.(w[1] * x .+ w[2])
    y2 = w[3] * y1 .+ w[4]
    return y2
end

result=create_vocab("wordcount.txt")
#loadcorpus("en-ud-dev.conllu",result)
wordcount=length(result) # size of feature vector
wordlist=Dict{String, Int}()
for word in keys(result)
    get!(wordlist,word,length(wordlist)+1)
end
#w=randn(4*fsize+68)
#fvector=Array{Float64}(4*fsize+68)
sentences,postaglists,parentlists=read_conllfile("en-ud-dev.conllu")
sentence_embeddings=[]
features=[]
gold_scores=[]
sentence_lengths=[]

for i=1:2#length(sentences)
#i=1
    n=length(sentences[i])
    #wembed=zeros(wordcount+1,n)
    wembeds=[]
    for j=1:n
        wembed=zeros(wordcount+1)
        if(haskey(wordlist, sentences[i][j]))
            wembed[wordlist[sentences[i][j]]]=1.0
        else
            wembed[end]=1.0
        end
        push!(wembeds,wembed)
    end
    arc_features=[]
    for j=1:n
        for k=1:n
            push!(arc_features,vcat(wembeds[j],wembeds[k]))
        end
    end
    lambda_gold=zeros(n+1,n+1)
    for j=1:length(parentlists[i])
        lambda_gold[parentlists[i][j]+1,j+1]=1
    end
    push!(sentence_embeddings, wembeds)
    push!(features,arc_features)
    push!(gold_scores,lambda_gold)
    push!(sentence_lengths,n)
end

weight=0.01*rand(length(features[1][1]))

#=for i=1:5#length(sentences)
    pred_scores=predict(weight, features[i], sentence_lengths[i])
    loss=mse(gold_scores[i],pred_scores)
end=#

#lambda=zeros(n+1,n+1)
optim=optimizers(weight,Adam,lr=0.001)

#word_scores=weight'*wembed

function predict(weight, arc_feat, n)
    lambda = Array{Any}(zeros(n+1,n+1))
    for j=1:n
        for k=1:n
            #println("$(j+1), $(k+1), $lambda[j+1,k+1]")
            lambda[j+1,k+1] = j==k ? 0:
            weight'*arc_feat[n*(j-1)+k]
        end
    end
    return lambda
end

function graph_loss(weight, arc_feat, n, lambda_gold#=, optim=#)
    lambda_pred = predict(weight, arc_feat, n)
    return mse(lambda_pred, lambda_gold)
    #return nll(lambda_pred, lambda_gold)
end

graph_loss_grad = grad(graph_loss)  #grads. shape is the same as weight.

function train!(weight, sentences, features, gold_scores, optim)
    for i=1:2#length(sentences)
        n=length(sentences[i])
        grads=graph_loss_grad(weight, features[i], n, gold_scores[i])
        update!(weight, grads, optim)
    end
end

lambda = predict(weight, features[1], sentence_lengths[1])
#Chu_Liu_Edmonds(sentences[1], lambda)
display(CKY(sentences[1], lambda))

test_sentence=["a","b","c"]
#=test_lambda=[0.0 0.0 0.0;
             1.0 0.0 1.0;
             0.0 0.0 0.0]=#
             
test_lambda=[0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 0.0;
             0.0 1.0 0.0 1.0;
             0.0 0.0 0.0 0.0]

#display(Eisner(test_sentence, test_lambda))

#=
    for j=1:n
        lambda[1,j+1]=1-word_scores[j]
    end
    for j=1:n
        for k=1:n
            lambda[j+1,k+1]=abs(word_scores[j]-word_scores[k])
        end
    end

    for j=1:n
        for k=1:n
            lambda[j+1,k+1]=abs(word_scores[j]-word_scores[k])
        end
    end
lambda_gold=zeros(n+1,n+1)
for j=1:length(parentlists[i])
    lambda_gold[parentlists[i][j]+1,j+1]=1
end
=#
    #display(CKY(sentences[i],lambda_gold))
#end

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
