<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

This is a user guide for the constraint function in pc-gym which will walkthrough an example of how to add a constraint to an environment.

### Constraint Definition
Firstly, a dictionary named constraints is defined. The keys in the dictionary represent the state, and the values represent the constraints on that state. In this case, the state '0' has no constraint, and the state '1' has a constraint of 332. Then a dictionary named constraint_type is defined. The keys in the dictionary represent the state, and the values represent the type of constraint on that state. In this case, the state '1' has a less than or equal to (<=) constraint. Mathematically, the constraint is represent by the following inequality.

\begin{align}
\nonumber x_1 \leq 332
\end{align}

Then the environment parameters are updated with these dictionaries and two addtional boolean variables: `done_on_cons_vio` which allows the episode to end if the constraint is violated and `r_penalty` which adds a penalty to the reward function for violating the constraint. 

```py
# Constraint Definition
constraints = {
    '0': None, # No constraint on the reactor concentration of species A
    '1' : 332 # Constraint on the reactor temperature
}

# Define the constraint type
constraint_type = {'1':'<='}

# Update the environment parameters
env_params.update({
'done_on_cons_vio':False, # Done on constraint violation
'constraints': constraints, 
'cons_type': constraint_type,
'r_penalty': True  # Add a penalty for constraint violation to the reward function
})
```

### Training with a Constraint Example 

A policy is now trained on the environment with a constraint. We use the <a href ="https://arxiv.org/abs/1707.06347"> Proximal Policy Optimization (PPO)</a> algorithm implemented by <a href="https://stable-baselines3.readthedocs.io/en/master/#">Stable Baselines 3</a>.
```py
Constraint_Policy = PPO('MlpPolicy', env, verbose=1,learning_rate=0.01).learn(total_timesteps=3e4)
env.plot_rollout(Constraint_Policy,10)   
```
<figure>
  <img src="../../img/Constraint Policy.png" alt="Image title" style="width:100%">
</figure>


### Custom Constraints