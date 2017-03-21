function Shift!(stack,buffer,arcset)
    push!(stack,buffer[1])
    print("Shift=> ")
    printTransition(stack,buffer[2:end],arcset)
end

function printTransition(stack,buffer,arcset)
    print("Stack: ")
    println(stack)
    print("Buffer: ")
    println(buffer)
    print("Arcset: ")
    println(arcset)
end

#dataset=readdlm("/mnt/ai/data/nlp/conll17/Universal Dependencies 2.0/ud-treebanks-conll2017/UD_English/en-ud-train.conllu";quotes=false)
dataset=readdlm("en-ud-train.conllu";quotes=false)

sent=[]
stack=[]
sptr=0
buffer=[]
reltype=[]
data_index=1
dataset_size=size(dataset)[1]
column_size=size(dataset)[2]
head=[]
input_vector=[]

for column_index=2:column_size
    for data_index=1:dataset_size
        if findfirst(input_vector,dataset[data_index,column_index])==0
            push!(input_vector,dataset[data_index,column_index])
        end
    end
end

x=zeros(Int,size(input_vector)[1],dataset_size)

for input_index=1:dataset_size
    for column_index=2:column_size
        find_index=find(input_vector.==dataset[input_index,column_index])
        if find_index!=0
            x[find_index,input_index]=1
        end
    end
end

while(data_index<dataset_size)
    if(typeof(dataset[data_index,1])==Int)
        if(dataset[data_index,1]<dataset[data_index+1,1])
            push!(buffer,dataset[data_index,1])
            push!(sent,string(dataset[data_index,2]))
            push!(reltype,dataset[data_index,4])
            push!(head,dataset[data_index,7])
        else
            #println(head)
            arcset=zeros(Int,length(buffer))
            while(!isempty(buffer))
                if(sptr>0)
	            if(head[stack[sptr]]==buffer[1])
                        arcset[stack[sptr]]=buffer[1]
                        pop!(stack)
                        sptr-=1
                        print("Left Arc=> ")
                        printTransition(stack,buffer,arcset)
                    elseif(head[buffer[1]]==stack[sptr])
                        right_arc=true
                        for w in find(head.==buffer[1])
                            if (arcset[w]!=buffer[1])
                                right_arc=false
                                break
                            end
                        end
                        if(right_arc)
                            arcset[buffer[1]]=stack[sptr]
                            buffer[1]=pop!(stack)
                            sptr-=1
                            print("Right Arc=> ")
                            printTransition(stack,buffer,arcset)
                        else
                            Shift!(stack,buffer,arcset)
                            sptr+=1
                            buffer=buffer[2:end]
                        end
                    else
                        Shift!(stack,buffer,arcset)
                        sptr+=1
                        buffer=buffer[2:end]
                    end
                else
                    Shift!(stack,buffer,arcset)
                    sptr+=1
                    buffer=buffer[2:end]
                end
            end
            sptr=0
            sent=[]
            stack=[]
            buffer=[]
            reltype=[]
            head=[]
        end
    end
    data_index+=1
end
