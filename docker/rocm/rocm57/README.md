##  Base mlc_llm docker image for ROCm 5.7 systems

Make sure you perform:

`sh ./buildimage.sh`

This will build the base docker image for ROCm 5.7, from the latest nightly.  The resulting image will be on your local registry, you can further push the image to any deployment registry.  The image size will be very large (about 28.1GB) since it includes all ROCm toolkit and support libraries,
