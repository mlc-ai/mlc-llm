(function(window, document) {
    // Test if it already supports srcset
    if ('srcset' in document.createElement('img'))
        return true;

    // We want to get the device pixel ratio
    var maxWidth   = (window.innerWidth > 0) ? window.innerWidth : screen.width,
        maxHeight  = (window.innerHeight > 0) ? window.innerHeight : screen.height,
        maxDensity = window.devicePixelRatio || 1;

    // Implement srcset
    function srcset(image) {
        if (!image.attributes['srcset']) return false;

        var candidates = image.attributes['srcset'].nodeValue.split(',');

        for (var i = 0; i < candidates.length; i++) {
            // The following regular expression was created based on the rules
            // in the srcset W3C specification available at:
            // http://www.w3.org/html/wg/drafts/srcset/w3c-srcset/

            var descriptors = candidates[i].match(
                    /^\s*([^\s]+)\s*(\s(\d+)w)?\s*(\s(\d+)h)?\s*(\s(\d+)x)?\s*$/
                ),
                filename = descriptors[1],
                width    = descriptors[3] || false,
                height   = descriptors[5] || false,
                density  = descriptors[7] || 1;

            if (width && width > maxWidth) {
                continue;
            }

            if (height && height > maxHeight) {
                continue;
            }

            if (density && density > maxDensity) {
                continue;
            }

            image.src = filename;
        }
    }


    var images = document.getElementsByTagName('img');

    for (var i=0; i < images.length; i++) {
        srcset(images[i]);
    }
})(window, document);