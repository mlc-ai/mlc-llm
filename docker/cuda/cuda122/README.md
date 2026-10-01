##  Base mlc_llm docker image for Cuda 12.2 systems

Make sure you perform:

`sh ./buildimage.sh`

This will build the base docker image for Cuda 12.2, from the latest nightly.  The resulting image will be on your local registry, you can further push the image to any deployment registry.  The image size will be very large (about 18.4GB) since it includes all cuda toolkit and support libraries,
