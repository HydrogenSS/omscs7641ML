import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt

def Frozen_Lake_Experiments():
	# 0 = left; 1 = down; 2 = right;  3 = up

	environment  = 'FrozenLake-v0'
	env = gym.make(environment)
	env = env.unwrapped
	desc = env.unwrapped.desc

	
	### POLICY ITERATION ####
	time_array_PI=[0]*10
	gamma_arr=[0]*10
	iters_PI=[0]*10
	list_scores_PI=[0]*10

	print('POLICY ITERATION WITH FROZEN LAKE')
	for i in range(0,10):
		st=time.time()
		best_policy,k = policy_iteration(env, gamma = (i+0.5)/10)
		scores = evaluate_policy(env, best_policy, gamma = (i+0.5)/10)
		end=time.time()
		gamma_arr[i]=(i+0.5)/10
		list_scores_PI[i]=np.mean(scores)
		iters_PI[i] = k
		time_array_PI[i]=end-st
	


	### VALUE ITERATION ###

	time_array_VI=[0]*10
	gamma_arr=[0]*10
	iters_VI=[0]*10
	list_scores_VI=[0]*10
	
	print('VALUE ITERATION WITH FROZEN LAKE')
	# fig = plt.figure()
	best_vals=[0]*10
	for i in range(0,10):
		st=time.time()
		best_value,k = value_iteration(env, gamma = (i+0.5)/10)
		policy = extract_policy(env,best_value, gamma = (i+0.5)/10)
		policy_score = evaluate_policy(env, policy, gamma=(i+0.5)/10, n=1000)
		gamma = (i+0.5)/10
		plot = plot_policy_map('Frozen-Lake-Policy-Map-VI-Gamma-'+str(gamma) ,policy.reshape(4,4),desc,colors_lake(),directions_lake())
		end=time.time()
		gamma_arr[i]=(i+0.5)/10
		iters_VI[i]=k
		best_vals[i] = best_value
		# plt.plot(gamma_arr,best_value,label=f'gamma={gamma_arr[i]}')
		list_scores_VI[i]=np.mean(policy_score)
		time_array_VI[i]=end-st

	plt.plot(gamma_arr,best_vals)
	plt.xlabel('Gammas')
	plt.ylabel('Optimal Value')
	plt.title('Frozen Lake - Value Iteration - Best Value Analysis')
	plt.legend()
	plt.grid()
	#plt.show()
	plt.savefig('FrozenLake-VI-bestValue.png')
	plt.close()



	##########  PLOT  #############

	plt.plot(gamma_arr, time_array_PI, label='PI')
	plt.plot(gamma_arr, time_array_VI, label='VI')
	plt.xlabel('Gammas')
	plt.title('Frozen Lake - Execution Time Analysis')
	plt.ylabel('Execution Time (s)')
	plt.legend()
	plt.grid()
	#plt.show()
	plt.savefig('FrozenLake-execTime.png')
	plt.close()

	plt.plot(gamma_arr,list_scores_PI, label='PI')
	plt.plot(gamma_arr,list_scores_VI, label='VI')
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title('Frozen Lake - Reward Analysis')
	plt.legend()
	plt.grid()
	#plt.show()
	plt.savefig('FrozenLake-reward.png')
	plt.close()

	plt.plot(gamma_arr,iters_PI, label='PI')
	plt.plot(gamma_arr,iters_VI, label='VI')
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title('Frozen Lake - Convergence Analysis')
	plt.legend()
	plt.grid()
	#plt.show()
	plt.savefig('FrozenLake-convergence.png')
	plt.close()

	#################################

	
	### Q-LEARNING #####
	print('Q LEARNING WITH FROZEN LAKE')
	# st = time.time()
	reward_array = []
	iter_array = []
	size_array = []
	chunks_array = []
	averages_array = []
	time_array = []
	Q_array = []
	for epsilon in [0.05,0.15,0.25,0.5,0.75,0.90]:
		initialeps = epsilon
		st = time.time()
		Q = np.zeros((env.observation_space.n, env.action_space.n))
		rewards = []
		iters = []
		optimal=[0]*env.observation_space.n
		alpha = 0.85
		gamma = 0.95
		episodes = 30000
		environment  = 'FrozenLake-v0'
		env = gym.make(environment)
		env = env.unwrapped
		desc = env.unwrapped.desc
		for episode in range(episodes):
			state = env.reset()
			done = False
			t_reward = 0
			max_steps = 1000000
			for i in range(max_steps):
				if done:
					break        
				current = state
				if np.random.rand() < (epsilon):
					action = np.argmax(Q[current, :])
				else:
					action = env.action_space.sample()
				
				state, reward, done, info = env.step(action)
				t_reward += reward
				Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
			epsilon=(1-2.71**(-episode/1000))
			rewards.append(t_reward)
			iters.append(i)


		for k in range(env.observation_space.n):
			optimal[k]=np.argmax(Q[k, :])

		print()
		print(epsilon)
		print(optimal)
		plot = plot_policy_map('Frozen-Lake-Policy-Map-Q-iniEps-'+ str(initialeps), np.array(optimal).reshape(4,4),desc,colors_lake(),directions_lake())

		reward_array.append(rewards)
		iter_array.append(iters)
		Q_array.append(Q)

		env.close()
		end=time.time()
		#print("time :",end-st)
		time_array.append(end-st)

		# Plot results
		def chunk_list(l, n):
			for i in range(0, len(l), n):
				yield l[i:i + n]

		size = int(episodes / 50)
		chunks = list(chunk_list(rewards, size))
		averages = [sum(chunk) / len(chunk) for chunk in chunks]
		size_array.append(size)
		chunks_array.append(chunks)
		averages_array.append(averages)

	plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0],label='epsilon=0.05')
	plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1],label='epsilon=0.15')
	plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2],label='epsilon=0.25')
	plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3],label='epsilon=0.50')
	plt.plot(range(0, len(reward_array[4]), size_array[4]), averages_array[4],label='epsilon=0.75')
	plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5],label='epsilon=0.95')
	plt.legend()
	plt.xlabel('Iterations')
	plt.grid()
	plt.title('Frozen Lake - Q Learning - Constant Epsilon')
	plt.ylabel('Average Reward')
	#plt.show()
	plt.savefig('FrozenLake-Q-constEpsilon.png')
	plt.close()

	plt.plot([0.05,0.15,0.25,0.5,0.75,0.95],time_array)
	plt.xlabel('Epsilon Values')
	plt.grid()
	plt.title('Frozen Lake - Q Learning')
	plt.ylabel('Execution Time (s)')
	#plt.show()
	plt.savefig('FrozenLake-Q-execTime.png')
	plt.close()


	# plt.subplot(1,6,1)
	plt.subplot(2,3,1)
	plt.imshow(Q_array[0])
	plt.title('Epsilon=0.05')

	# plt.subplot(1,6,2)
	plt.subplot(2,3,2)
	plt.title('Epsilon=0.15')
	plt.imshow(Q_array[1])

	# plt.subplot(1,6,3)
	plt.subplot(2,3,3)
	plt.title('Epsilon=0.25')
	plt.imshow(Q_array[2])

	# plt.subplot(1,6,4)
	plt.subplot(2,3,4)
	plt.title('Epsilon=0.50')
	plt.imshow(Q_array[3])

	# plt.subplot(1,6,5)
	plt.subplot(2,3,5)
	plt.title('Epsilon=0.75')
	plt.imshow(Q_array[4])

	# plt.subplot(1,6,6)
	plt.subplot(2,3,6)
	plt.title('Epsilon=0.95')
	plt.imshow(Q_array[5])

	# plt.subplot(1,7,7)
	# plt.colorbar()

	#plt.show()
	plt.savefig('FrozenLake-Qarray.png')
	plt.close()
### end Frozen_Lake_Experiments() #########

def run_episode(env, policy, gamma, render = True):
	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs, reward, done , _ = env.step(int(policy[obs]))
		total_reward += (gamma ** step_idx * reward)
		step_idx += 1
		if done:
			break
	return total_reward

def evaluate_policy(env, policy, gamma , n = 100):
	scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
	return np.mean(scores)

def extract_policy(env,v, gamma):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
		policy[s] = np.argmax(q_sa)
	return policy

def compute_policy_v(env, policy, gamma):
	v = np.zeros(env.nS)
	eps = 1e-5
	while True:
		prev_v = np.copy(v)
		for s in range(env.nS):
			policy_a = policy[s]
			v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
		if (np.sum((np.fabs(prev_v - v))) <= eps):
			break
	return v

def policy_iteration(env, gamma):
	policy = np.random.choice(env.nA, size=(env.nS))  
	max_iters = 200000
	desc = env.unwrapped.desc
	# print('desc')
	# print(desc)
	# print()
	# print('policy')
	# print(policy.shape)
	# print(policy)
	# print()
	for i in range(max_iters):
		old_policy_v = compute_policy_v(env, policy, gamma)
		new_policy = extract_policy(env,old_policy_v, gamma)
		# print(old_policy_v.shape) ################
		# print(new_policy.shape) ##############
		# print(new_policy) ##############
		if i % 2 == 0:
			plot = plot_policy_map('Frozen-Lake-Policy-Map-PI'+ '-Gamma-' + str(gamma)+ '-Iteration-'+ str(i) ,new_policy.reshape(4,4),desc,colors_lake(),directions_lake())
			# plot = plot_policy_map('Taxi-Policy-Map-Iteration-'+ str(i) + '-(Policy Iteration)-'+ '-Gamma-' + str(gamma),new_policy.reshape(4,5,25),desc,colors_taxi(),actions_taxi())
			a = 1
		if (np.all(policy == new_policy)):
			k=i+1
			break
		policy = new_policy
	return policy,k

def value_iteration(env, gamma):
	v = np.zeros(env.nS)  # initialize value-function
	max_iters = 100000
	eps = 1e-20
	desc = env.unwrapped.desc
	for i in range(max_iters):
		prev_v = np.copy(v)
		for s in range(env.nS):
			q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
			v[s] = max(q_sa)
		if i % 50 == 0:
			plot = plot_policy_map('Frozen-Lake-Policy-Map-VI-Gamma-'+ str(gamma)+'-Iteration-'+ str(i),v.reshape(4,4),desc,colors_lake(),directions_lake())
		if (np.sum(np.fabs(prev_v - v)) <= eps):
			k=i+1
			break
	return v,k

def plot_policy_map(title, policy, map_desc, color_map, direction_map):
	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
	font_size = 'x-large'
	if policy.shape[1] > 16:
		font_size = 'small'
	plt.title(title)
	# print(policy.shape)
	# print(policy)
	# print(map_desc.shape)
	for i in range(policy.shape[0]):
		for j in range(policy.shape[1]):
			y = policy.shape[0] - i - 1
			x = j
			p = plt.Rectangle([x, y], 1, 1)
			p.set_facecolor(color_map[map_desc[i,j]])
			ax.add_patch(p)
			text = ax.text(x+0.5, y+0.5, direction_map[int(policy[i, j])], weight='bold', size=font_size, horizontalalignment='center', verticalalignment='center', color='w')
			

	plt.axis('off')
	plt.xlim((0, policy.shape[1]))
	plt.ylim((0, policy.shape[0]))
	plt.tight_layout()
	# print(title)
	# print(policy)
	# print()
	plt.savefig(title+str('.png'))
	plt.close()

	return (plt)





def Forest_Experiments():
	# import mdptoolbox, mdptoolbox.example
	import pandas as pd
	# import mdptoolbox-hiive
	from hiive import mdptoolbox
	import hiive.mdptoolbox.example
	import hiive.mdptoolbox.mdp

	P, R = mdptoolbox.example.forest(S=500,r1=40,r2=20,p=0.1)
	# P, R = mdptoolbox.example.forest()
	# print(P)
	# print(R)
	# print()
	time_array_3 = []
	value_f_3 = []
	iters_3 = []

	
	print('POLICY ITERATION WITH FOREST MANAGEMENT')
	
	value_f = [0]*10
	value_all = [0]*10
	policy = [0]*10
	iters = [0]*10
	time_array = [0]*10
	gamma_arr = [0] * 10

	# gamma_arr = [0.05,0.15,0.25,0.5,0.65,0.75,0.85,0.95]
	for i in range(0,10):

		tic = time.perf_counter()
		pi = mdptoolbox.mdp.PolicyIteration(P, R, (i+0.5)/10)
		pi.run()
		toc = time.perf_counter()
		gamma_arr[i]=(i+0.5)/10
		value_f[i] = np.mean(pi.V)
		value_all[i] = pi.V
		# print(pi.V)
		policy[i] = pi.policy
		# print(np.array(pi.policy))
		iters[i] = pi.iter
		# time_array[i] = pi.time
		time_array[i] = toc-tic

	policydf = pd.DataFrame(np.array(policy))
	print(np.array(policy))
	policydf.to_csv('forest-PI-policy.csv')
	valuealldf = pd.DataFrame(np.array(value_all))
	# print(np.array(value_all))
	valuealldf.to_csv('forest-PI-valueall.csv')


	time_array_3.append(time_array)
	value_f_3.append(value_f)
	iters_3.append(iters)

	# plt.plot(gamma_arr,iters)
	# plt.xlabel('Gammas')
	# plt.ylabel('Iterations to Converge')
	# plt.title('Forest Management - Policy Iteration - Convergence Analysis')
	# plt.grid()
	# # plt.show()
	# plt.savefig('ForestManagement-PI-convergence.png')
	# plt.close()

	print('VALUE ITERATION WITH FOREST MANAGEMENT')
	# P, R = mdptoolbox.example.forest()
	value_f = [0]*10
	value_all = [0]*10
	policy = [0]*10
	iters = [0]*10
	time_array = [0]*10
	gamma_arr = [0] * 10
	for i in range(0,10):

		tic = time.perf_counter()
		vi = mdptoolbox.mdp.ValueIteration(P, R, (i+0.5)/10)
		vi.run()
		toc = time.perf_counter()
		gamma_arr[i]=(i+0.5)/10
		value_f[i] = np.mean(vi.V)
		value_all[i] = vi.V
		policy[i] = vi.policy
		iters[i] = vi.iter
		# time_array[i] = vi.time
		time_array[i] = toc-tic

	policy_expected = vi.policy # the highest discount rate

	policydf = pd.DataFrame(np.array(policy))
	print(np.array(policy))
	policydf.to_csv('forest-VI-policy.csv')
	valuealldf = pd.DataFrame(np.array(value_all))
	# print(np.array(value_all))
	valuealldf.to_csv('forest-VI-valueall.csv')



	time_array_3.append(time_array)
	value_f_3.append(value_f)
	iters_3.append(iters)

	# plt.plot(gamma_arr,iters)
	# plt.xlabel('Gammas')
	# plt.ylabel('Iterations to Converge')
	# plt.title('Forest Management - Value Iteration - Convergence Analysis')
	# plt.grid()
	# # plt.show()
	# plt.savefig('ForestManagement-VI-convergence.png')
	# plt.close()
	
	
	print('Q LEARNING WITH FOREST MANAGEMENT')

	tic = time.perf_counter()

	# qlearn = mdptoolbox.mdp.QLearning(P,R,0.95)
	qlearn = mdptoolbox.mdp.QLearning(P,R,0.95,alpha_decay=0.99,epsilon_decay=0.999)
	toc = time.perf_counter()
	print(toc-tic)
	qlearn.run()
	# print('Q matrix',qlearn.Q)
	print('value-function',qlearn.V)
	print('policy',qlearn.policy)
	# print(qlearn.run_stats)
	# print(qlearn.mean_discrepancy)
	run_stats = qlearn.run_stats
	iters = [ rs['Iteration'] for rs in run_stats ]
	errors = [ rs['Error'] for rs in run_stats ]
	alphas = [ rs['Alpha'] for rs in run_stats ]
	epsilons = [ rs['Epsilon'] for rs in run_stats ]
	# errors = [ run_stats[i]['Error'] for i in range(len(run_stats)) ]

	# P, R = mdptoolbox.example.forest(S=2000,p=0.01)
	value_f = []
	value_all = []
	policy = []
	time_array = []
	Q_table = []
	error_all = []
	mean_v_all = []
	for gamma in gamma_arr: # discount rate
		tic = time.perf_counter()
		qlearn = mdptoolbox.mdp.QLearning(P,R,gamma,
			alpha_decay=0.99,
			epsilon_decay=0.999,
			n_iter=80000)
		qlearn.run()
		toc = time.perf_counter()
		value_f.append(np.mean(qlearn.V))
		value_all.append(qlearn.V)
		policy.append(qlearn.policy)
		time_array.append(toc-tic)
		error_all.append(qlearn.run_stats[-1]['Error'])
		mean_v_all.append(qlearn.run_stats[-1]['Mean V'])


	policydf = pd.DataFrame(np.array(policy))
	print(np.array(policy))
	policydf.to_csv('forest-Q-policy.csv')
	valuealldf = pd.DataFrame(np.array(value_all))
	# print(np.array(value_all))
	valuealldf.to_csv('forest-Q-valueall.csv')
	


	time_array_3.append(time_array)
	value_f_3.append(value_f)

	plt.plot(gamma_arr,error_all)
	plt.xlabel('Gammas')
	plt.ylabel('errors')
	plt.title('Forest Management - Q Learning - Error')
	plt.grid()
	# plt.show()
	plt.savefig('ForestManagement-Q-error.png')
	plt.close()

	plt.plot(gamma_arr,mean_v_all)
	plt.xlabel('Gammas')
	plt.ylabel('Mean Value Function')
	plt.title('Forest Management - Q Learning - Mean Value')
	plt.grid()
	# plt.show()
	plt.savefig('ForestManagement-Q-meanV.png')
	plt.close()


	################# PLOT #######################
	print()
	print('...plotting time & reward & iters')

	plt.plot(gamma_arr, time_array_3[0], label='PI')
	plt.plot(gamma_arr, time_array_3[1], label='VI')
	# plt.plot(gamma_arr, time_array_3[2], label='Qlearn')
	plt.xlabel('Gammas')
	plt.title('Forest Management - Execution Time ')
	plt.ylabel('Execution Time (s)')
	plt.legend()
	plt.grid()
	#plt.show()
	plt.savefig('ForestManagement-execTime-PIVI.png')
	plt.close()

	plt.plot(gamma_arr, value_f_3[0], label='PI')
	plt.plot(gamma_arr, value_f_3[1], label='VI')
	plt.plot(gamma_arr, value_f_3[2], label='Qlearn')
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title('Forest Management - Reward Analysis')
	plt.legend()
	plt.grid()
	#plt.show()
	plt.savefig('ForestManagement-reward.png')
	plt.close()


	plt.plot(gamma_arr,iters_3[0], label='PI')
	plt.plot(gamma_arr,iters_3[1], label='VI')
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title('Forest Management - Convergence Analysis')
	plt.grid()
	# plt.show()
	plt.savefig('ForestManagement-convergence.png')
	plt.close()

	########################################

	return
### end Forest_Experiments() #########



def colors_lake():
	return {
		b'S': 'green',
		b'F': 'skyblue',
		b'H': 'black',
		b'G': 'gold',
	}

def directions_lake():
	return {
		3: '⬆',
		2: '➡',
		1: '⬇',
		0: '⬅'
	}


import random
random.seed(318)
print('STARTING EXPERIMENTS')
# Frozen_Lake_Experiments()
Forest_Experiments()
print('END OF EXPERIMENTS')




