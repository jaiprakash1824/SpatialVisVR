# SpatialVisVR: A Virtual Reality Framework for Spatial Data Visualization

This repository encompasses MainApis, searchUIApis, the search_engine_algorithm, and a compressed Unity project complete with an apk file. To effectively deploy and test the project, follow the step-by-step guide provided below.

### Execution Steps:

1. **Network Connectivity:** Ensure all devices are connected to the same Wi-Fi network.
   
2. **Environment Setup:** Install the Python environment specified in the `requirements.txt` file.
   
3. **Launching the APIs:** Run `app1.py` from both the MainApis and searchUIApis directories in separate terminal instances.
   
4. **Unity Project Configuration:** 
   - Navigate to the Unity project directory.
   - Update the API endpoint with the appropriate IP address.
   - Subsequently, build the application and deploy it to the VR headset.
   
5. **Search Execution (Optional):** 
   - To initiate a search, launch the Droid Cam application on your smartphone. 
   - Execute `PathologyPipeline.py` from the `search_engine_algorithm` directory. This activates the phone's camera.
   - Focus the camera on the desired slide.
   - In the VR application, tap the "update" button to synchronize and display the search results.

By adhering to this guide, users can seamlessly visualize spatial data in a virtual reality environment.