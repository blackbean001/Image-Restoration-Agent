**Intro**    
1. This repo contains an agent system for images restoration with mixed degradation.
2. The repo is inspired by AgenticIR (https://github.com/Kaiwen-Zhu/AgenticIR) with following improvements:
   (1) We rebuilt the end-to-end pipeline with LangGraph for better states management and efficiency.
   (2) We replaced offline model inference with service-based inference to be suitable for production-level deployment.
   (3) We use a ServiceManager to manage the model services. Least used service will be released when target GPU utilization is beyond some threshold.
   (4) We try to accelerate the restoration process by using CLIP4CIR (https://github.com/ABaldrati/CLIP4Cir/tree/master) to find similar images in the database.

**AgenticIR**:   
1. Run "docker build ." to create environment. One can use "conda env list" to check env for different models.
2. Run "sh synthesize.sh" to generate synthesized low-quality data (to train CLIP4CIR).
3. Run "python -m pipeline.infer" to generate restoration results and saved in output. One can first use evaluate_degradation_by="depictqa" to generate initial outputs. After saving enough knowledge in Step 4, one can use evaluate_degradation_by="clip_retrieval" for efficienty.
4. Refer to AgenticIR/retrieval_database/CLIP4CIR/run_pipeline.sh to train the model for image quality classification and insert history knowledge to PostgreSQL.

**AgentApp**:   
1. Use LangGraph to reproduce the functionality of AgenticIR, making it easier to manage the pipeline. To add new tools, one can easily define new nodes and link edges to existing nodes in the graph. Run 'run.sh' to test inference.  
3. Service enabled by FastAPI, run 'test_api.sh' to test.

**To-dos**:   
1. Use a service manager to kill least-used services when overloaded. (done).
2. Adaptively select GPU rank when launching new service for better GPU utilization.
3. Support for Kubernetes deployment.
4. Use GPU pooling (like TensorFusion <https://github.com/NexusGPU/tensor-fusion>), MPS or time-slicing to improve GPU utilization rate. 
5. Accelerate inference speed for individual models.


![image](https://github.com/blackbean001/Auto-Image-Restoration/blob/main/pngs/pipeline.png)
