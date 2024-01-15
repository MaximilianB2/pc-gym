# Policy Evaluation Class for pc-gym 
import numpy as np
import matplotlib.pyplot as plt




class policy_eval():
    '''
    Policy Evaluation Class 

    Inputs: Environment, policy and number of policy repitions

    Outputs: Plots of states/control/constraints/setpoints (complete),
             return distribution (incomplete), expected return (incomplete), 
             oracle trajectories (incomplete) and lower confidence bounds (incomplete)
    '''
    def __init__(self,Models_env,policy,reps,env_params):
        self.env = Models_env(env_params) 
        self.policy = policy 
        self.reps = reps
        
   
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
        states = np.zeros((self.env.x0.shape[0], self.env.N))
        actions = np.zeros((self.env.action_space.low.shape[0], self.env.N))

        o, _ = self.env.reset()
        for i in range(self.env.N):
            a, _states = self.policy.predict(o)
            o, r, term, trunc, info = self.env.step(a)
            actions[:, i] = (a + 1)*(self.env.env_params['a_space']['high'] - self.env.env_params['a_space']['low'])/2 + self.env.env_params['a_space']['low']
            states[:, i] = (o + 1)*(self.env.env_params['o_space']['high'] - self.env.env_params['o_space']['low'])/2 + self.env.env_params['o_space']['low']
            total_reward += r

        return total_reward, states, actions

    def plot_rollout(self):
            '''
            Function to plot the rollout of the policy

            Inputs:
                policy - policy to be rolled out
                reps - number of rollouts to be performed

            Outputs:
                Plot of states and actions with setpoints and constraints if they exist]
            
            '''
            states = np.zeros((self.env.x0.shape[0],self.env.N,self.reps))
            actions = np.zeros((self.env.Nu,self.env.N,self.reps))
            rew = np.zeros((self.env.N,self.reps))
            for r_i in range(self.reps):
                rew[:,r_i], states[:,:,r_i], actions[:,:,r_i] = self.rollout()
            t = np.linspace(0,self.env.tsim,self.env.N)
            len_d = 0
            if self.env.disturbance_active:
                len_d = len(self.env.disturbances)

            plt.figure(figsize=(10, 2*(self.env.Nx+self.env.Nu+len_d)))
            for i in range(self.env.Nx):
                plt.subplot(self.env.Nx + self.env.Nu+len_d,1,i+1)
                plt.plot(t, np.median(states[i,:,:],axis=1), 'r-', lw=3,label = 'x_' + str(i))
                plt.gca().fill_between(t, np.min(states[i,:,:],axis=1), np.max(states[i,:,:],axis=1), 
                                color='r', alpha=0.2 )
                if str(i) in self.env.SP:
                    plt.plot(t, self.env.SP[str(i)], color = 'black', linestyle = '--', label='Set Point')
                if self.env.constraint_active:
                    if str(i) in self.env.constraints:
                        plt.hlines(self.env.constraints[str(i)], 0,self.env.tsim,'r',label='Constraint')
                plt.ylabel('x_'+str(i))
                plt.xlabel('Time (min)')
                plt.legend(loc='best')
                plt.xlim(min(t), max(t))

            for j in range(self.env.Nu-len_d):
                plt.subplot(self.env.Nx+self.env.Nu+len_d,1,j+self.env.Nx+1)
                plt.step(t, np.median(actions[j,:,:],axis=1), 'b--', lw=3, label='u_'+str(j))
                plt.ylabel('u_'+str(j))
                plt.xlabel('Time (min)')
                plt.legend(loc='best')
                plt.xlim(min(t), max(t))

            if self.env.disturbance_active:       
                for k in self.env.disturbances.keys():
                    if self.env.disturbances[k].any() != None:
                        i=0
                        plt.subplot(self.env.Nx+self.env.Nu+len_d,1,i+self.env.Nx+self.env.Nu-len_d+1)
                        plt.plot(t, self.env.disturbances[k],"r",label='d_'+str(i))
                        plt.xlabel('Time (min)')
                        plt.ylabel('d_'+str(i))
                        plt.xlim(min(t), max(t))
                        i+=1

            plt.tight_layout()
            plt.show()