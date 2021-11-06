# Set up GCloud: 
Google sponsors this course with free GCloud credits through the Cloud Higher Education Programs.

Go to [this link](https://gcp.secure.force.com/GCPEDU?cid=HGfyYSrJnwOmvOkly5FUWbeGvTTyaYsuzI%2BUJ3owt3WKC%2BSJW%2BoIZzdGozvQAzWe/) to claim your coupon code for the credits associated with this course.  
Once you have your coupon code go to [this link](https://console.cloud.google.com/education?authuser=4) to get your credits (you will need a Google account, if needed, you can create one using your INSA mail address).
Once you are on the GCloud homepage, start by creating a new project:  

![](img/gcloud/gcloud0.png)
![](img/gcloud/gcloud01.png)  
Once your project is created go to Compute Engine -> VM instances
![](img/gcloud/gcloud02.png) and activate Compute Engine API for your project ![](img/gcloud/gcloud03.png).  
You now need to add GPU capacity to your project.  
Go to IAM and admin -> Quotas  
![](img/gcloud/gcloud04.png).  
On the filter select Quota ![](img/gcloud/gcloud05.png) and GPUs(all regions) ![](img/gcloud/gcloud06.png) and click on modify quotas ![](img/gcloud/gcloud07.png).  
Increase the limit to at least one and add a short description to your request.  ![](img/gcloud/gcloud08.png)
This process may take some time. Therefore, be sure sure to complete every step at least a few days before the practical session.

You will also need to install the Cloud SDK Command-line interface.
It should already be installed on the INSA's machines.
If you are using your personal computer, follow the corresponding installation procedure available [here](https://cloud.google.com/sdk/docs/install).

