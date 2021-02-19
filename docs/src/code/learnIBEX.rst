:github_url: https://github.com/guochengqian/deep_learning_phd_wiki

.. _use_ibex:

How to use IBEX 
=================

IBEX is the GPU cluster hosted in KAUST. IBEX is managed using Slurm. This Documentation teachs you how to use Slurm GPU cluster. IBEX is just an example. You can use what mentions in this doc in any Slurm based cluster (like the node center in IVUL-KAUST, the GPU clusters in most companies). 


Termius
--------

Recall that when you connect to a cluster (or a remote machine), you have to ``ssh yourAccount@ipAddress``, and then input your password. This is annoying because you have to do it everytime when you want to login in the cluster.
 
To avoid inputing username and password every time, I highly recommend a terminal software called `termius`_ to all of you.

After installing termius, simply add the host (the your username in termius:  add address of the cluster or any remote machine -> click ssh -> add username and password.
Note: the address of the host can be the actual IP or the ip label, like in IBEX, you can use ``glogin.kaust.edu.sa`` or the actual IP Address, I recommend using the IP address)

After adding the host, whenever you want to login in host, you just need to click the host name

For example, setting a host named (glogin) for logging gloin in IBEX:

-  address: 10.109.66.127 (or, you can use ``glogin.ibex.kaust.edu.sa``)
-  username: qiang (this is my username, please use yours)
-  password: xxxxxxxx

After setting up, only a click on the glogin button in termius is needed to ssh into the ibex system.


Tmux
------
``tmux`` is very helpful if you want to connect to a remote machine.
If you do not use ``tmux``, you will lose your connection if your terminal somehow disconnects with your remote machine (this situation happens a lot).
tmux is easy to use. You only have to ``tmux new -s xxx`` to create a new tmux window. You can then login into the window using ``tmux a -t xxx``.
Then even if you lose your connection with the remote machine, you can re-login, and the tmux is always there. You can use tmux to login into that windown, and you will find everything is continues running there without being affected by the disconnection.
There is a `tmux cheatsheet`_. Here is the `tmux wiki <https://github.com/tmux/tmux/wiki>`_.


Run a Job in cluster (IBEX)
-----------------------------------
In any Slurm based cluster, whenever you want to use any resources (node, CPU, GPU, Mem, and running Time), you should ask a request for the resources. 

You can either use ``sbatch`` or ``srun`` to request for resources to run your program in cluster.

1. sbatch
    The first option to run a program (\eg, train or evaluate a deep learning model) is to use sbatch to send your job. Sbatch send your job in the priority queue and your code will continue to run even if you lose connection with cluster for some reason.

    There is an example of sbatch file:

    ::

       #!/bin/bash --login
       #SBATCH -N 1
       ##SBATCH --array=1-5  # uncomment to repeat the task, \eg, if you want to run the same program for 5 times, you can use --array=1-5
       #SBATCH -J rloc
       #SBATCH -o slurm_log/%x.%3a.%A.out    # make sure the slurm_log folder exists
       #SBATCH -e slurm_log/%x.%3a.%A.err
       #SBATCH --time=5-0:00:00 # time, you'd better make job less than 24 hours therefore you get resources faster. You can split your job into smalle chunks if the total time is over 24 hours.
       #SBATCH --gpus=8            # or: --gres=gpu:v100:8 to specify the GPUs.
       ##SBATCH --constraint=[rtx2080ti|gtx1080ti|v100] # or use this line to specify the GPUS. Note: you either use this line and above line to specify GPUs, or you use single line: --gres=gpu:v100:8
       #SBATCH --gpus-per-node=8   # use gpu_wide. specify the node, only consider node with 8 GPUs. comment out if you do not need this constraint.
       #SBATCH --cpus-per-gpu=6 # cps-per-gpu can not be over 6, for fair use
       #SBATCH --mem-per-gpu=45G # can not be over 45G for fair use
       #SBATCH --mail-user=xxx@kaust.edu.sa    # send message to your email
       #SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90
       ##SBATCH -A conf-gpu-2020.11.23 # if you have additional QOS (that can give you higher priority for requesting resources, you can add it)
       #SBATCH --constraint=[ref_32T]  # use shared folder in v100s. this line can only be added if you request for 8 v100s node, and you want to use datasets in shared folder.

       # activate your conda env
       echo "Loading anaconda..."

       module purge
       module load gcc
       module load cuda/10.1.105

       conda activate deepgcn

       echo "...Anaconda env loaded"
       python -u examples/classification/train.py  --phase train --train_path /scratch/dragon/intel/lig/guocheng/data/deepgcn/modelnet40
       echo "...training function Done"

    Run ``sbatch train_ibex.sh`` then your job will be put in the queue. You can check your queue info by : ``squeue -u yourusername``.
    You can check your job output and error through the file ``slurm_log/%x.%3a.%A.out`` and ``slurm_log/%x.%3a.%A.err``.

    You can check the available GPUs by: ``gpu-usage --nodes|grep v100``

    See `KAUST IBEX offical doc`_ for detailed information. See the `IBEX
    Best Practice`_ for the detailed configuration of best usage on IBEX.

2. srun
    srun allow you to use cluster just like in terminal on your local
    machine. This is very useful when you want to debug your code.
    srun is convenient to use, however it will stop run when you
    lose connection with ibex. You need tmux to protect the node. When
    you lose connection, you can use tmux to login back into the node.

    You can also srun into your allocated node using:
    ``srun --jobid=yourjobid --pty bash``
    To do that, you have to use Sbatch at first to query for resources and
    start your training there. The srun is just used as a tube. After you
    srun into the node, you can check your mem usage using ``nvidia-smi``, etc.


Use NodeCenter (IVUL-KAUST)
-----------------------------------
If you are the member of IVUL-KAUST, you can use the internal GPU cluster belonging to our group called NodeCenter.
The usage of the NodeCenter is roughly the same of IBEX.
However, you have to login into each node and request resources in each node.

Note: the time limit for a job is 12:00:00, so you must cut your job into small jobs (chunks) if the whole time is over 12 hours.


.. _termius: https://termius.com/
.. _KAUST IBEX offical doc: https://www.hpc.kaust.edu.sa/sites/default/files/files/public/Cluster_training/26_11_2018/0_Ibex_cheat_sheet_Nov_26_2018.pdf
.. _IBEX Best Practice: ../../files/Deep%20Learning%20Best%20Practices.pdf
.. _tmux cheatsheet: https://gist.github.com/MohamedAlaa/2961058
