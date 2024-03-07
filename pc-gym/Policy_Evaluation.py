# Policy Evaluation Class for pc-gym
import numpy as np
import matplotlib.pyplot as plt
from oracle import oracle

class policy_eval():
    '''
    Policy Evaluation Class

    Inputs: Environment, policy and number of policy repitions

    Outputs: Plots of states/control/constraints/setpoints (complete),
             return distribution (incomplete), expected return (incomplete),
             oracle trajectories (incomplete) and lower confidence bounds (incomplete)
    '''
    def __init__(self,Models_env,policy,reps,env_params, oracle = False, MPC_params = False):
        self.Models_env = Models_env
        self.env_params = env_params
        self.env = Models_env(env_params)
        
        self.policy = policy
        self.reps = reps
        self.oracle = oracle
 
        self.MPC_params  = MPC_params  
        

    def rollout(self):
        '''
        Rollout the policy for N steps and return the total reward, states and actions

        Input:
            policy - policy to be rolled out

        Outputs:
            total_reward - total reward obtained
            states - states obtained from rollout
            actions - actions obtained from rollout

        '''
        
        total_reward = 0
        s_rollout = np.zeros((self.env.x0.shape[0], self.env.N))
        actions = np.zeros((self.env.env_params['a_space']['low'].shape[0], self.env.N))
        
        o, r = self.env.reset()
        total_reward = r['r_init']
        s_rollout[:,0] = (o + 1)*(self.env.env_params['o_space']['high'] - self.env.env_params['o_space']['low'])/2 + self.env.env_params['o_space']['low']
        for i in range(self.env.N-1):
            a, _s = self.policy.predict(o)
            
            o, r, term, trunc, info = self.env.step(a)
            
            actions[:, i] = (a + 1)*(self.env.env_params['a_space']['high'] - self.env.env_params['a_space']['low'])/2 + self.env.env_params['a_space']['low']
            s_rollout[:, i+1] = (o + 1)*(self.env.env_params['o_space']['high'] - self.env.env_params['o_space']['low'])/2 + self.env.env_params['o_space']['low']
            total_reward += r

        if self.env.constraint_active:
            cons_info = info['cons_info']
        else:
            cons_info = np.zeros((1,self.env.N,1))
        a, _s = self.policy.predict(o)
        actions[:,self.env.N-1] = (a + 1)*(self.env.env_params['a_space']['high'] - self.env.env_params['a_space']['low'])/2 + self.env.env_params['a_space']['low']
        
        return total_reward, s_rollout, actions,cons_info
    
    def plot_rollout(self, reward_dist = False, cons_viol = False):
        '''
        Function to plot the rollout of the policy

        Inputs:
            policy - policy to be rolled out
            reps - number of rollouts to be performed

        Outputs:
            Plot of states and actions with setpoints and constraints if they exist]

        '''
        data = {}
        action_space_shape = self.env.env_params['a_space']['low'].shape[0]
        num_states = self.env.x0.shape[0]
        
        states = np.zeros((num_states, self.env.N, self.reps))
        actions = np.zeros((action_space_shape, self.env.N, self.reps))
        rew = np.zeros((1, self.reps))

        # Collect Oracle data
        if self.oracle:
            r_opt = np.zeros((1,self.reps))
            x_opt = np.zeros((self.env.Nx, self.env.N, self.reps))
            u_opt = np.zeros((self.env.Nu, self.env.N, self.reps))
            oracle_instance = oracle(self.Models_env, self.env_params,self.MPC_params)
            for i in range(self.reps):
                x_opt[:, :, i], u_opt[:, :, i] = oracle_instance.mpc()
                for k in self.env.SP:
                    state_i = self.env.model.info()['states'].index(k)
                    r_opt[:,i] += np.sum((x_opt[state_i,:,i] - self.env.SP[k])**2)*-1*self.env_params['r_scale'][k] 
            data.update({'r_opt':r_opt,'x_opt':x_opt,'u_opt':u_opt})   
     
        try:
            cons_info = np.zeros((self.env.n_con,self.env.N,1,self.reps))
        except:
            cons_info = np.zeros((1,self.env.N,1,self.reps))

        # Collect RL rollouts
        for r_i in range(self.reps):
            rew[:, r_i], states[:, :, r_i], actions[:, :, r_i], cons_info[:,:,:,r_i]= self.rollout()
        data.update({'r_RL':rew,'x_RL':states,'u_RL':actions})
        if self.env.constraint_active:
            data.update({'cons_viol_RL':cons_info})

        t = np.linspace(0, self.env.tsim, self.env.N)
        len_d = 0

        if self.env.disturbance_active:
            len_d = len(self.env.model.info()['disturbances'])

        plt.figure(figsize=(10, 2*(self.env.Nx+self.env.Nu+len_d)))
        for i in range(self.env.Nx):
            plt.subplot(self.env.Nx + self.env.Nu+len_d,1,i+1)
            plt.plot(t, np.median(states[i,:,:],axis=1), color='tab:red', lw=3,label = self.env.model.info()['states'][i])
            plt.gca().fill_between(t, np.min(states[i,:,:],axis=1), np.max(states[i,:,:],axis=1), color='tab:red', alpha=0.2, edgecolor = 'none')
            if self.oracle:
                plt.plot(t, np.median(x_opt[i,:,:],axis=1), color='tab:blue', lw=3,label = 'Oracle '+ self.env.model.info()['states'][i])
                plt.gca().fill_between(t, np.min(x_opt[i,:,:],axis=1), np.max(x_opt[i,:,:],axis=1), color='tab:blue', alpha=0.2,edgecolor = 'none' )
            if self.env.model.info()['states'][i] in self.env.SP:
                plt.step(t, self.env.SP[self.env.model.info()['states'][i]],where = 'post', color = 'black', linestyle = '--', label='Set Point')
            if self.env.constraint_active:
                if self.env.model.info()['states'][i] in self.env.constraints:
                    plt.hlines(self.env.constraints[self.env.model.info()['states'][i]], 0, self.env.tsim, color = 'black',label='Constraint')
            plt.ylabel(self.env.model.info()['states'][i])
            plt.xlabel('Time (min)')
            plt.legend(loc='best')
            plt.grid('True')
            plt.xlim(min(t), max(t))

        for j in range(self.env.Nu-len_d):
            plt.subplot(self.env.Nx+self.env.Nu+len_d,1,j+self.env.Nx+1)
            plt.step(t, np.median(actions[j,:,:],axis=1), color='tab:red', lw=3, label=self.env.model.info()['inputs'][j])
            if self.oracle:
                plt.step(t, np.median(u_opt[j,:,:],axis=1), color='tab:blue', lw=3, label='Oracle '+ str(self.env.model.info()['inputs'][j]))
            if self.env.constraint_active:
                for con_i in self.env.constraints:
                    if self.env.model.info()['inputs'][j] == con_i:
                        plt.hlines(self.env.constraints[self.env.model.info()['inputs'][j]], 0,self.env.tsim,'r',label='Constraint')
            plt.ylabel(self.env.model.info()['inputs'][j])
            plt.xlabel('Time (min)')
            plt.legend(loc='best')
            plt.grid('True')
            plt.xlim(min(t), max(t))

        if self.env.disturbance_active:
            for i, k in enumerate(self.env.disturbances.keys()):
                if self.env.disturbances[k].any() is not None:
                    plt.subplot(self.env.Nx+self.env.Nu+len_d,1,i+self.env.Nx+self.env.Nu-len_d+1)
                    plt.step(t, self.env.disturbances[k],"r",label=k)
                    plt.xlabel('Time (min)')
                    plt.ylabel(k)
                    plt.xlim(min(t), max(t))
        plt.tight_layout()
        plt.show()
        if cons_viol:
            plt.figure(figsize=(12, 3 * self.env.n_con))
            con_i = 0
            for i, con in enumerate(self.env.constraints):
                for j in range(len(self.env.constraints[str(con)])):
                    plt.subplot(self.env.n_con,1,con_i+1)
                    plt.title(f'{con} Constraint')
                    plt.step(t, np.sum(cons_info[con_i,:,:,:],axis=2), color = 'tab:green', label = f'{con} Violation (Sum over Repetitions)')
                    plt.grid('True')
                    plt.xlabel('Time (min)')
                    plt.ylabel(con)
                    plt.xlim(min(t), max(t))
                    plt.legend(loc = 'best')
                    con_i += 1
            plt.tight_layout()
            plt.show()

        if reward_dist:
            min_val = min(r_opt.min(), rew.min())
            max_val = max(r_opt.max(), rew.max())
            bins = np.linspace(min_val, max_val, 50)  

            plt.figure(figsize=(12, 8))  
            plt.grid(True, linestyle='--', alpha=0.6)  

            if self.oracle:
                plt.hist(r_opt.flatten(), bins=bins, color='tab:blue', alpha=0.5, label='Oracle', edgecolor='black')  
            plt.hist(rew.flatten(), bins=bins, color='tab:red', alpha=0.5, label='RL Algorithm', edgecolor='black')  

            plt.xlabel('Return', fontsize=14)  
            plt.ylabel('Frequency', fontsize=14)  
            plt.title('Distribution of Expected Return', fontsize=16)  
            plt.legend(fontsize=12)  

            plt.show()

        return data

    
