# CVAT guide

How to label a small seed dataset in [CVAT](https://app.cvat.ai/) and export it in **YOLO 1.1** format for this repo.

## 1. Create a new task

Open the **Tasks** page and click the blue **+** button.

![Create new task](CVAT_guide_images/add%20new%20task.png)

Choose **Create a new task**.

![Create a new task menu](CVAT_guide_images/add%20new%20task_2.png)

## 2. Name the task and upload images

Enter a task name, then upload your images (`.jpg`, `.jpeg`, `.png`) from **My computer**.

![Basic configuration — name and upload files](CVAT_guide_images/basic%20config.png)

## 3. Add your class label(s)

In the **Constructor** tab, type your class name (for example `fish`, `cat`, …) and add it.  
These names become `obj.names` when you export.

![Add label name](CVAT_guide_images/add%20lable%20name.png)

## 4. Submit the task

Click **Submit & Open**.

![Submit and Open](CVAT_guide_images/submit%20and%20open.png)

You land on the task page.

![Task page](CVAT_guide_images/task%20opens.png)

## 5. Open the job

Under **Jobs**, click the job link to open the annotation editor.

![Open the job](CVAT_guide_images/open%20job.png)

## 6. Draw bounding boxes

The annotation workspace opens on your first image.

![Annotation workspace](CVAT_guide_images/job%20opens.png)

Select **Draw new rectangle**, pick your label, use **By 2 points**, then click **Shape**.

![Draw new rectangle](CVAT_guide_images/create%20BB.png)

Click two corners around each object. Adjust the box if needed, then **Save**. Move through all frames and label them.

![Bounding box created](CVAT_guide_images/BB%20created.png)

## 7. Export as YOLO 1.1

Go back to **Tasks**, open the task **Actions** menu (⋮), and choose **Export task dataset**.

![Export task dataset](CVAT_guide_images/back%20to%20task%20to%20export%20the%20BB.png)

Set:

- **Export format:** `YOLO 1.1`
- **Save images:** turn **on** (so images are included with the labels)
- Optional custom zip name

Click **OK**.

![Export format YOLO 1.1](CVAT_guide_images/chooes%20format.png)

Wait until the export finishes, then download the zip.

![Export in progress](CVAT_guide_images/exporting.png)

## 8. What you get

After unzipping, you should see something like:

![Downloaded YOLO folder](CVAT_guide_images/downloded%20folder%20.png)

| File / folder | What it is |
|---|---|
| `obj.names` | Class names (one per line) |
| `obj.data` | Dataset config from CVAT |
| `train.txt` | List of image paths |
| `obj_train_data/` | Images + matching `.txt` labels |

Each label file looks like this (YOLO format):

`class_id x_center y_center width height` (values normalized 0–1)

![Example label .txt](CVAT_guide_images/downloded%20BB%20cor.png)

## 9. Copy into this repo

Put the export here:

```
data/dataset/
  obj.names
  obj.data          # optional
  train.txt         # optional
  obj_train_data/   # .jpg/.png images + matching .txt labels
```

Then continue with training in the main [README](README.md).
