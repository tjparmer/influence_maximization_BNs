#RBN and game theory computations
#Thomas Parmer, 2022

import random
import networkx as nx
import time
from cana.boolean_node import BooleanNode
import cana
from simulations import *
from driver_sets import *


#generate look-up table based on theta (if there at least as many 1's as theta, output is 1)
#linear threshold model; shortcut for strategies where response is based on a threshold
def output_transitions_LT(n,node,theta):
    """ Given a networkx graph n and node, generate a LUT for node based on theta threshold """
    total=2**len(n[node]) #total combinations to try
    output_list=[]
    for i in range(total):
        trial_string = statenum_to_binstate(i,len(n[node])) #from utils.py in cana
        #print trial_string
        #simply count the 1's to determine if the threshold is met
        if trial_string.count('1')/len(trial_string) >= theta: 
            output_list.append(1)
        else:
            output_list.append(0)
    return output_list


#generate look-up table randomly based on p
def output_transitions_rnd(n,node,p):
    """ Given a networkx graph n and node, generate a LUT for node with random outputs biased by p """
    total=2**len(n[node]) #total combinations to try
    output_list=[]
    for i in range(total):
        trial_string = statenum_to_binstate(i,len(n[node])) #from utils.py in cana
        #print trial_string
        #random selection
        if random.random() < p:
            output_list.append(0)
        else:
            output_list.append(1)
    return output_list


#generate output transitions given a payoff matrix and a player's strategy
#NOTE: state 0 is defection, state 1 is cooperation
def output_transitions_GT(n,node,payoff,strategy="br"):
        """ Given a networkx graph n and node, generate a LUT for node based on a payoff matrix and strategy,
        by default the strategy is 'br' for best response; 
        payoff matrices should be in the form of payoff value per state: 00,01,10,11 for two players """
        total=2**len(n[node]) #total combinations to try
        output_list=[]
        for i in range(total):
            trial_string = statenum_to_binstate(i,len(n[node])) #from utils.py in cana
            #print trial_string
            #check the payoff matrix if either strategy was played
            payoff0=trial_string.count('0')*payoff[0] + trial_string.count('1')*payoff[1] #node defects
            payoff1=trial_string.count('0')*payoff[2] + trial_string.count('1')*payoff[3] #node cooperates
            if strategy=='br':
                if payoff1>=payoff0: #cooperation chosen in tie scenario
                    action=1
                else:
                    action=0
            elif strategy=='tft': #tit-for-tat, choose majority response of your neighbors
                if trial_string.count('1')>=trial_string.count('0'):
                    action=1
                else:
                    action=0
            #print action       
            output_list.append(action)
        return output_list


#determine the inputs of a BooleanNode, graph, or digraph by checking the structural graph
from cana.boolean_node import BooleanNode
def predecessors(node,N):
    """ takes a BooleanNode and networkx (di)graph as inputs """
    if isinstance(node,BooleanNode):
        node=node.name
    if isinstance(N,nx.DiGraph):
        return N.predecessors(node)
    return N[node] #otherwise it's all of our neighbors


#determine the outgoing transitions of a BooleanNode by checking the structural graph
from cana.boolean_node import BooleanNode
def neighbors(node,N):
    """ takes a BooleanNode and networkx (di)graph as inputs """
    if isinstance(node,BooleanNode):
        node=node.name
    return N[node]


#create BooleanNetwork object by converting all output transitions relative to theta, requires logic dictionary
#NOTE: BooleanNetworks must be 0-based
#G = nx.Graph([('a','b'),('b','c')]) does not work unless nodes 'a', 'b' given integer names
def convert_GT(N,payoff,strategy='br',subtract=False):
    """ takes a network and global threshold value (theta) and generates a BooleanNetwork object with LUTs based on theta """
    
    #logic = {i:{} for i in range(len(N.nodes()))}
    logic = {i:{} for i in N.nodes()}
    
    for node in N:
        if subtract: i=int(node)-1
        else: i=int(node)
        logic[i] = {'name': node, 'in':[], 'out':[]} 
        if subtract:
            logic[i]['in']=[int(key)-1 for key in predecessors(node,N)] #assumes nodes with integer names!, must decrease by 1 to make 0-based
        else:
            logic[i]['in']=[int(key) for key in predecessors(node,N)]
        logic[i]['out']=output_transitions_GT(N,node,payoff,strategy)
        #print node,N.degree([node]),len(logic[i]['out']),logic[i]['in']

    #N = cana.BooleanNetwork(name='', logic=logic, Nnodes=len(N), constants={}, keep_constants=True)
    return cana.BooleanNetwork.from_dict(logic)


#create BooleanNetwork object by converting all output transitions randomly
#NOTE: BooleanNetworks must be 0-based
#G = nx.Graph([('a','b'),('b','c')]) does not work unless nodes 'a', 'b' given integer names
def convert_rnd(N,p=0.5,subtract=False):
    """ takes a network and global activation probability (p) and generates a BooleanNetwork object with LUTs based on p """
    
    #logic = {i:{} for i in range(len(N.nodes()))}
    logic = {i:{} for i in N.nodes()}
    
    for node in N:
        if subtract: i=int(node)-1
        else: i=int(node)
        logic[i] = {'name': node, 'in':[], 'out':[]} 
        if subtract:
            logic[i]['in']=[int(key)-1 for key in predecessors(node,N)] #assumes nodes with integer names!, must decrease by 1 to make 0-based
        else:
            logic[i]['in']=[int(key) for key in predecessors(node,N)]
        logic[i]['out']=output_transitions_rnd(N,node,p)
        #print node,N.degree([node]),len(logic[i]['out']),logic[i]['in'],sum(logic[i]['out'])/len(logic[i]['out'])

    #N = cana.BooleanNetwork(name='', logic=logic, Nnodes=len(N), constants={}, keep_constants=True)
    return cana.BooleanNetwork.from_dict(logic)


#function to evaluate an output transition; NOTE: inputs and outputs are strings
def output_transition(node,inputs,method='literal',act_theta=0.5):
    """ determine which, if any, output transition can occur; depends on the method;
    by default, the method is 'literal' (all input states need to be known)
    other methods include 'threshold', where input activation above theta triggers an output of 1 """
    
    inputs=''.join([ip if ip else '#' for ip in inputs])
    #print node.inputs,inputs
    output=False
    if method=='literal': #all inputs must be present
        if len([ip for ip in inputs if ip!='#'])==len(node.inputs): #we have enough information to update the state
            output=str(node.outputs[binstate_to_statenum(inputs)])
    elif method=='threshold': #checks to see if simple threshold is met
        if inputs.count('1')/float(len(inputs))>=act_theta:
            output='1'
        elif inputs.count('0')/float(len(inputs))>(1-act_theta):
            output='0'
    
    return output


#general BFS to work with a discrete dynamical system on a network; expects a BooleanNetwork object
#in contrast to a DCM, the given network has variable nodes and not s-units
#NOTE: all variable names are strings; the original interaction graph N is needed for easy traversal
#NOTE: the original graph N assumes integer variable names unless names='string'
def BFS_general(n,seed,N,input_type='steady',time_limit=1,pinned=set([]),method='literal',
                act_theta=0.5,names='int'):
    
    if not isinstance(seed,list): #convert seed to a list if it is a single node
        seed=[seed]
        
    #construct s-units from variable nodes, NOTE: assumes binary for now
    nodes={str(node.name)+'-0' for node in n.nodes}
    nodes=nodes.union({str(node.name)+'-1' for node in n.nodes})
    node_map={str(node.name):node for node in n.nodes} #assumes nodes have unique names
    
    for node in seed:
        if node not in nodes:
            raise Exception('Seed '+str(node)+' not in the network!  Hint: did you specify both the node and current state? e.g. wg-1')
            
    #create the initial state of the network
    nt_state={str(node.name): False for node in n.nodes}
    for node in seed:
        nt_state[node[:-2]]=node[-1] #assumes less than 10 states; otherwise string length will get messed up!
    for node in pinned: nt_state[node[:-2]]=node[-1]
    update_state={} #hold all updates for this time step
    #print len(nodes),nt_state
            
    #if there is steady-state input (pinning control), note any contradictions to any node in the seed
    #NOTE: if given a logical contradiction, this function picks only ONE of the states to use
    #however, this doesn't affect steady-state (because the variable was already added to the queue and active step)
    if input_type=='steady':
        #vals={n.node[x]['variable']:n.node[x]['state'] for x in seed if n.node[x]['type']=='variable'}
        constants={node[:-2] for node in seed}
    elif pinned: #check pinned nodes as well
        #vals={n.node[x]['variable']:n.node[x]['state'] for x in pinned if n.node[x]['type']=='variable'}
        constants={node[:-2] for node in pinned} 
    else: constants={}
    contradictory_nodes={node for node in nodes if node[:-2] in constants and 
                        nt_state[node[:-2]] and node[-1]!=nt_state[node[:-2]]}
    #print contradictory_nodes   
    counter=0  #this will increment towards the time limit and provide an exit if we get stuck
    visited=set() #visited nodes during the entire history
    #thresholds={} #global threshold dictionary to populate with discounted threshold nodes
        
    #create 'priority queue', actually a dictionary of time steps due to difficulty in changing priorities in a heap
    time_steps={0: deque(seed)} #FIFO queue for each time step of nodes to attempt to visit
    time=0 #global chronometer, iterates by one as dynamic unfolding takes place
    unfolding={} #track the unfolding of the dynamics over iterated time, mirrors time_steps but only holds visited nodes    
    active_step={node[:-2] for node in seed} #active nodes for this time step, will be different than visited if input_type='pulse'
    for node in pinned:
        time_steps[0].append(node)
        active_step.add(node[:-2])
    
    
    while time<=max(time_steps): #allows for possibility of time jumps, i.e. updates further ahead than one time step
        
        #update counter
        old_length=len(visited)
        visited.update(active_step)
        #if len(visited)==old_length: #increase counter if we have not added anything new
        #if any(unfolding[t] == active_step for t in unfolding): #increase counter if we repeat a past partial configuration
        counter+=1
        #print time,len(visited)
        #reset thresholds and active_step if this is pulse (so that we may revisit nodes, including the starting nodes)
        #print time,active_step
        if input_type=='pulse':
            #thresholds={}
            active_step=set()
            #add in pinned nodes first
            for node in pinned:
                time_steps.setdefault(time+1,deque())
                time_steps[time+1].append(node)
                active_step.add(node[:-2])
        
        #check for exit due to being stuck in a cycle
        if counter>=time_limit: 
            break

        #print {key:nt_state[key] for key in nt_state if nt_state[key]}
        #run BFS
        while time_steps.get(time): 
            node=time_steps[time].popleft()
            #add node to unfolding
            unfolding.setdefault(time,set())
            unfolding[time].add(node) #adding s-unit
            #print node
            
            #check neighbors - note that for steady state we assume all past actors are still activated, but pulse requires repeat activation
            if names=='string': var=node[:-2]
            else: var=int(node[:-2])
            for neighbor in neighbors(var,N): #NOTE: assumes that original nodes are integers
                neighbor=str(neighbor)
                if neighbor not in active_step:
                    #check that neighbor is able to update
                    inputs=[nt_state[str(ip)] for ip in node_map[neighbor].inputs] #if nt_state[str(ip)]]
                    output=output_transition(node_map[neighbor],inputs,method=method,act_theta=act_theta)
                    #print neighbor,node_map[neighbor].inputs,''.join([ip if ip else '#' for ip in inputs]),output
                    if output: #we have enough information to update the state
                        #at this point we have passed the update rule, so insert neighbor at proper time slot t
                        t = time + 1 #n.node[neighbor]['delay']
                        output=neighbor+'-'+output
                        #ignore any contradictory nodes; better to ignore than remove because we don't want BFS to be destructive to the graph
                        if output in contradictory_nodes:
                            continue
                        time_steps.setdefault(t,deque())
                        time_steps[t].append(output)
                        active_step.add(output[:-2])
                        update_state[output[:-2]]=output[-1]
        
        #next iteration
        if input_type=='pulse':
            nt_state={str(node.name): False for node in n.nodes} #reset all states except those that are pinned 
            for node in pinned: nt_state[node[:-2]]=node[-1]
        #add updates from this round
        for key in update_state: nt_state[key]=update_state[key]
        update_state={} #hold all updates for the next time step
        time+=1
    
    return unfolding


#run a random full-network configuration based on a single seed of one or more specified nodes
#TODO: update simulations to run on brute-force calculation rather than BFS_general
def run_simulation(NB,seed,N,input_type='pulse',time_limit=1,pinned={},method='literal',act_theta=0.5,unknown_prob=0.5,
                   runs=100,iterations=10,nodes={},names='int'):
    """ run various simulations of a single seed, while assuming other nodes have unknown_prob chance of activation 
    returns the generated simulations; NOTE: this depends on BFS_general
    iterations refers to the number of iterations past the initial configuration at t=0 (iterations+1 total) 
    NOTE: seed must be a list of s-units """

    iterations=iterations+1 #include 0
    #time_limit=iterations+1 #can modify later
    if not nodes:
        nodes={str(node.name) for node in NB.nodes}
    act_prob_sim = {i:{node: 0.0 for node in nodes} for i in range(iterations)}
    simulations={}
    if input_type=='pulse' and time_limit<iterations+1:
        print("Time limit must be greater than the number of iterations by at least 2")
        return

    #get the ground truth for the seed based on several runs
    for i in range(runs):
        #initiate a random condition
        alt_seed=[sunit for sunit in seed]
        for node in nodes-{sunit[:-2] for sunit in alt_seed}:
            if random.random()<unknown_prob:
                alt_seed.append(node+'-1')
            else:
                alt_seed.append(node+'-0')
        #print alt_seed
        
        #find the information diffusion for the sample based on the BFS general algorithm
        simulations[str(alt_seed)]={i:{node: 0 for node in nodes} for i in range(iterations)} #to calculate a baseline
        diffusion=BFS_general(NB,alt_seed,N,input_type,time_limit,pinned,method,act_theta,names=names) #pin seed for steady-state modules
        nodes_reached=set([])
        #print diffusion

        for index in range(iterations):
            if input_type=='steady-state':
                if index < len(diffusion): #otherwise, nodes_reached is already at maximum
                    nodes_reached={node for node in diffusion[index]}
                for node in nodes_reached:
                    if node[-1]=='1': #only record activation probability (as inactivation is the converse of this)
                        act_prob_sim[index][node[:-2]]+=1
                        simulations[str(alt_seed)][index][node[:-2]]=1
            elif input_type=='pulse': #all states *should be in diffusion; time_limit must be > iterations
                for node in act_prob_sim[index]:
                    if node+'-1' in diffusion[index]:
                        act_prob_sim[index][node]+=1
                        simulations[str(alt_seed)][index][node]=1
                    elif node+'-0' in diffusion[index]: #0 is the default state so no need to set it
                        pass
                    else: #this applies to isolated nodes, for example; take last known state by default
                        act_prob_sim[index][node]+=simulations[str(alt_seed)][index-1][node]
                        simulations[str(alt_seed)][index][node]=simulations[str(alt_seed)][index-1][node]
        #print act_prob_sim[index][:-2]
            
    #find average (our label)
    act_prob_sim={index:{node:act_prob_sim[index][node]/runs for node in act_prob_sim[index]} for index in range(iterations)}
    #print seed,act_prob_sim[iterations-1]
    
    return act_prob_sim,simulations


#run random full-network configurations using a BFS-based graph approach
def run_simulations(NB,seeds,N,sunit_map,translator,input_type='pulse',pinned={},method='literal',length=None,runs=100,
                    act_theta=0.5,iterations=10,unknown_prob=0.5,nodes={},stats={},results=True,act_prob_sim={},simulations={},names='int'):
    """ run various simulations of the seeds, while assuming other nodes have unknown_prob chance of activation 
    prints out metrics and returns the generated simulations 
    iterations refers to the number of iterations past the initial configuration at t=0 (iterations+1 total)
    NOTE: this depends on BFS_general
    NOTE: pinned may either be 'seeds' or a set of s-units """
    
    if not nodes:
        nodes={str(node.name) for node in NB.nodes}
    total=0
    
    for seed in seeds:
        sunits=to_list(seed,sunit_map,translator)
        #print seed,sunits
        if length and len(sunits)!=length:
            continue
        if len({sunit[:-2] for sunit in sunits})!=len(sunits): #there is a contradiction
            continue
        total+=1
        if pinned=='seeds': #pin each seed
            pset={s for s in sunits}
        else: pset={s for s in pinned}
        act_prob_sim[str(seed)],simulations[str(seed)]=run_simulation(NB,sunits,N,input_type=input_type,time_limit=iterations+2,
            pinned=pset,method=method,act_theta=act_theta,unknown_prob=unknown_prob,runs=runs,iterations=iterations,nodes=nodes,names=names)
        #sunits=to_list(str(seed),dsunit_map,dtranslator)
        #act_prob_sim[str(seed)] = {i:{node: 0.0 for node in nodes} for i in range(iterations)}
        #simulations[str(seed)]={}
    if results:
        print('total seeds:'+str(total))
    return act_prob_sim,simulations


#find the baseline: compare each simulation against all other simulations per seed
#assumes that the diffusions from the simulations are stored in the simulations dictionary
#TODO: merge with compare_baseline (this just gets rid of the contradiction check)
def compare_baseline_RBNs(N,seeds,simulations,tau=0.0,iterations=10,stats={},results=True,verbose=False):
    """ compares a single simulation with all simulations of the same seed, prints out metrics and returns stats dictionary 
    NOTE: this applies only to Boolean networks """
    
    act_prob_base={seed:{} for seed in seeds}
    iterations=iterations+1 #include 0
    tps,fps=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    tns,fns=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    accuracy,recall,precision=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    errors,avg_sim=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)] #average similarity based on jaccard index
    total=0
    seed_total=0
    nodes={str(node.name) for node in N.nodes}
    for seed in seeds: 
        runs=len(simulations[seed]) #these should all be the same
        seed_total+=1
        
        #compare each run against all others to get the baseline
        for run in simulations[seed]:
            total+=1
            
            act_prob_base[seed][run] = {i:{node: 0.0 for node in nodes} for i in range(iterations)}
            for i in range(iterations):
                for node in nodes:
                    act_prob_base[seed][run][i][node]=sum([simulations[seed][run2][i][node] for run2 in simulations[seed] if run!=run2])/(runs-1)        
            #print act_prob_base[seed][run]

            #compare to our prediction
            for i in range(iterations):
                error=0.0 #error based on all node probabilities
                tp,fp,tn,fn=0.0,0.0,0.0,0.0 #keep track of statistics per iteration, based on nodes active within tau
                on_constants={node for node in act_prob_base[seed][run][i] if act_prob_base[seed][run][i][node]==1.0} #our constants (nodes always active)
                off_constants={node for node in act_prob_base[seed][run][i] if act_prob_base[seed][run][i][node]==0.0} #our constants (nodes always inactive)
                predicted_on={node for node in act_prob_base[seed][run][i] if simulations[seed][run][i][node]>=(1-tau)}
                predicted_off={node for node in act_prob_base[seed][run][i] if simulations[seed][run][i][node]<=(tau)}
            
                for node in act_prob_base[seed][run][i]:
                    label=act_prob_base[seed][run][i][node]
                    error+=abs(label-simulations[seed][run][i][node])**2
                    #stats based on actual module, NOTE: task is constant vs non-constant
                    if label==1 and simulations[seed][run][i][node]>=(1-tau) or label==0 and simulations[seed][run][i][node]<=(tau):
                        tp+=1
                    elif label==1 or label==0:
                        fn+=1
                    elif simulations[seed][run][i][node]>=(1-tau) or simulations[seed][run][i][node]<=(tau):
                        fp+=1
                    else:
                        tn+=1
        
                intersection=len(on_constants.intersection(predicted_on))+len(off_constants.intersection(predicted_off))
                union=len(on_constants.union(predicted_on))+len(off_constants.union(predicted_off))
                avg_sim[i]+=intersection/union
                #errors[i]+=math.sqrt(error)/len(act_prob_base[seed][run][i]) #normalize per node
                errors[i]+=error/len(act_prob_base[seed][run][i]) #FIXED FORMULA
                #print tp,fn,fp,tn,errors[i]
                #print seed,act_prob_base[seed][run][i]
                
                if on_constants or off_constants:
                    tps[i]+=tp/(len(on_constants)+len(off_constants))
                    fns[i]+=fn/(len(on_constants)+len(off_constants))
                else: #no true positives
                    tps[i]+=1
                other_set=nodes-on_constants-off_constants
                if other_set:
                    fps[i]+=fp/len(other_set)
                    tns[i]+=tn/len(other_set)
                else: #no true negatives
                    tns[i]+=1
                
                #summary statistics
                accuracy[i]+=(tp+tn)/(tp+tn+fp+fn)
                if (tp+fn)!=0:
                    recall[i]+=(tp)/(tp+fn)
                else: #no positives in module
                    recall[i]+=1
                if (tp+fp)!=0:
                    precision[i]+=(tp)/(tp+fp)
                else: #no positive predictions in module
                    precision[i]+=1
    
    if verbose: print(total,'total runs,',seed_total,'seeds')
    #normalize and print
    stats['tps']=[round(tps[i]/total,2) for i in range(iterations)]
    stats['fns']=[round(fns[i]/total,2) for i in range(iterations)]
    stats['fps']=[round(fps[i]/total,2) for i in range(iterations)]
    stats['tns']=[round(tns[i]/total,2) for i in range(iterations)]
    stats['accuracy']=[round(accuracy[i]/total,2) for i in range(iterations)]
    stats['recall']=[round(recall[i]/total,2) for i in range(iterations)]
    stats['precision']=[round(precision[i]/total,2) for i in range(iterations)]
    stats['avg_sim']=[round(avg_sim[i]/total,2) for i in range(iterations)]
    stats['errors']=[round(errors[i]/total,3) for i in range(iterations)]
    if results:
        print_stats(stats,seed_total)
        
    return stats


#for a given RBN model with fixed k, finds the driver set
def RBN_driver_selection(n=10,max_s=10,top=1,reduced=False,pinning={},tau=0.0,iterations=10,unknown_prob=0.5,
    t=10,base=2,normalized=True,pin_start=True,force=False,k=3,p=0.5,dcm=False):
    """ creates a RBN with fixed k; k MUST be at least 2; p is the bias towards 0 (1-activation prob)
    return the driver set if possible based on the top selection greedy heuristic"""
    
    #create the network, assumes k>=2
    if k==2:
        N = nx.cycle_graph(n).to_directed() #directed ring structure, ensures one connected component
    else:
        N = nx.random_regular_graph(int(k),n) #NOTE: this is undirected and isn't guaranteed to be connected
    
    #define the model
    N = convert_rnd(N,p)
    if reduced:
        ds={node.name: LUT_reduce(look_up_table(node)) for node in N.nodes}
    else:
        ds=None
    modules,translator={},{}
    sunits,sunit_map=get_sunits(N)
    
    #run the selection
    data={'size':0,'resolved':0,'time':0}
    start_time = time.time()
    selection=driver_selection(N,sunits,sunit_map,modules,translator,seeds=None,max_s=max_s,top=top,
        reduced=reduced,ds=ds,pinning=pinning,tau=tau,iterations=iterations,unknown_prob=unknown_prob,
        t=t,base=base,normalized=normalized,pin_start=pin_start,force=force,attractor=None,attractors=None,names='int')
    if selection is not None: #otherwise one was not found in the timeframe
        data['time']=time.time()-start_time
        data['size']=len(selection)
        if dcm:
            seed=to_list(selection,sunit_map,translator)
            DCM=N.dynamics_canalization_map(output=None, )
            #TODO: make this dynamic based off of parameters chosen above (steady or pulse)
            diffusion=BFS_threshold(DCM,seed,'steady',time_limit=iterations+2,pinned={})
            data['resolved']=count_sunits(diffusion,pinned=False)
            
    return selection,data


#generate statistics for RBNs of a given n and fixed k
def RBN_driver_selections(N=[],models=10,max_s=10,top=1,reduced=False,pinning={},tau=0.0,iterations=10,
    unknown_prob=0.5,t=10,base=2,normalized=True,pin_start=True,force=False,k=3,p=0.5,dcm=False,stats={}):
    """ generates several RBNs with fixed k over a range of n; k MUST be at least 2
    return statistics on average driver sets based on the top selection greedy heuristic"""
    
    for n in N:
        stats.setdefault(n,{'size':0.0,'time':0.0,'resolved':0.0})
        total=0
        for i in range(models):
            selection,data=RBN_driver_selection(n=n,max_s=max_s,top=top,reduced=reduced,pinning=pinning,
                tau=tau,iterations=iterations,unknown_prob=unknown_prob,t=t,base=base,normalized=normalized,
                pin_start=pin_start,force=force,k=k,p=p,dcm=dcm)
            if selection is None:
                print('no selection for n=',n,' and k=',k)
            else:
                total+=1
                stats[n]['size']+=data['size']
                stats[n]['time']+=data['time']
                stats[n]['resolved']+=data['resolved']
        if total:
            stats[n]['size']/=total
            stats[n]['time']/=total
            stats[n]['resolved']/=total
                
    return stats


#for a given RBN model with fixed k, finds the error of the mean-field approximation
def RBN_mf_error(n=100,reduced=False,pinning={},tau=0.0,iterations=10,unknown_prob=0.5,s=0,seeds=None,
    pin_start=True,k=1,p=0.5,samples=None,runs=100,act_theta=0.5,method='literal',baselines=True,verbose=True,names='int'):
    """ creates a RBN with fixed k; k may be 1,2, or higher; p is the bias towards 0 (1-activation prob)
    return the error of the mf-approximation for all seeds of size s"""
    
    #create the network
    if k==1 or k==2:
        NB = nx.cycle_graph(n).to_directed() #directed ring structure, ensures one connected component
    else:
        NB = nx.random_regular_graph(int(k),n) #NOTE: this is undirected and isn't guaranteed to be connected
        
    if k==1:  #remove excess edges from a directed graph (for k=1 ring structure)
        NB.remove_edge(0,n-1)
        for node in NB:
            if (node,node-1) in NB.edges(): NB.remove_edge(node,node-1)
    #print 'edges: ',len(NB.edges())
    
    #define the model
    NKB = convert_rnd(NB,p)
    if reduced:
        ds={node.name: LUT_reduce(look_up_table(node)) for node in NKB.nodes}
    else:
        ds=None
    modules,translator={},{}
    sunits,sunit_map=get_sunits(NKB)
    
    #find all modules of length s to compare against
    modules,translator=find_modules(NKB,s,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,
            tau=tau,iterations=iterations,pinning=pinning,p=p,seeds=seeds,data=True,samples=samples,verbose=verbose) 
    seeds=modules.keys()
    
    #create simulations
    act_prob_sim,simulations={},{}
    act_prob_sim,simulations=run_simulations(NKB,seeds,NB,sunit_map,translator,length=s,runs=runs,
        act_theta=act_theta,iterations=iterations,unknown_prob=unknown_prob,stats={},results=False,
        input_type='pulse',pinned="seeds",method=method,act_prob_sim=act_prob_sim,simulations=simulations,names=names)

    #compare simulations to the mean-field approximation
    stats=compare_sim_to_modules(NKB,seeds,modules,act_prob_sim,tau=tau,iterations=iterations,stats={},results=False)

    #compare to baselines
    bstats=compare_baseline_RBNs(NKB,seeds,simulations,tau=tau,iterations=iterations,stats={},results=False,verbose=verbose)

    return stats,bstats
   
