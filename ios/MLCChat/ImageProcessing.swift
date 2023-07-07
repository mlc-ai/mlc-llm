//
//  ImageProcessing.swift
//  MLCChat
//
//  Created by Kathryn Chen on 7/8/23.
//

import Foundation
import SwiftUI
import UIKit

// adapted from Mohammad Azam: https://github.com/azamsharp/SwiftUICamera
// delegate task to the coordinator to produce the image
struct ImagePicker : UIViewControllerRepresentable {
    typealias UIViewControllerType = UIImagePickerController
    typealias Coordinator = ImagePickerCoordinator

    @Binding var image: UIImage?
    @Binding var showImagePicker: Bool
    var imageSourceType: UIImagePickerController.SourceType = .photoLibrary

    func makeCoordinator() -> ImagePicker.Coordinator {
        return ImagePickerCoordinator(image: $image, showImagePicker: $showImagePicker)
    }

    func makeUIViewController(context: UIViewControllerRepresentableContext<ImagePicker>) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = imageSourceType
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: UIViewControllerRepresentableContext<ImagePicker>) {}
}

// image picker coordinator handling selecting from library or taking a photo
class ImagePickerCoordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    @Binding var image: UIImage?
    @Binding var showImagePicker: Bool

    init(image: Binding<UIImage?>, showImagePicker: Binding<Bool>) {
        _image = image
        _showImagePicker = showImagePicker
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let optionalImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            image = optionalImage
            showImagePicker = false
        }
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        showImagePicker = false
    }
}

// resize the input image to given width and height
func resizeImage(image: UIImage, width: Int, height: Int) -> UIImage {
    let shape = CGSize(width: width, height: height)
    UIGraphicsBeginImageContextWithOptions(shape, true, 0.0)
    image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    let resizedImage: UIImage? = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
    return resizedImage ?? image
}

// get the bit map of the image with normalization
func getBitMapImage(image: UIImage) -> UnsafeMutablePointer<Float16> {
    guard let cgImage = image.cgImage,
        let data = cgImage.dataProvider?.data,
        let bytes = CFDataGetBytePtr(data) else {
        fatalError("Couldn't access image data")
    }
    assert(cgImage.colorSpace?.model == .rgb)

    // allocate unsafe buffer pointer to create contiguous memory
    let buffer = UnsafeMutableBufferPointer<Float16>.allocate(capacity: 3 * cgImage.width * cgImage.height)

    // parameters for normalization
    let r_mean = 0.48145466
    let g_mean = 0.4578275
    let b_mean = 0.40821073
    let r_std = 0.26862954
    let g_std = 0.26130258
    let b_std = 0.27577711

    let bytesPerPixel = cgImage.bitsPerPixel / cgImage.bitsPerComponent
    var i = 0
    for idx in 0 ..< 2 {
        for y in 0 ..< cgImage.height {
            for x in 0 ..< cgImage.width {
                let offset = (y * cgImage.bytesPerRow) + (x * bytesPerPixel)
                var new_pixel = Float16(0.0)
                if idx == 0 {
                    new_pixel = Float16((Double(bytes[offset]) / 255.0 - r_mean) / r_std)
                } else if idx == 1 {
                    new_pixel = Float16((Double(bytes[offset + 1]) / 255.0 - g_mean) / g_std)
                } else if idx == 2 {
                    new_pixel = Float16((Double(bytes[offset + 2]) / 255.0 - b_mean) / b_std)
                }
                buffer[i] = new_pixel
                i += 1
            }
        }
    }

    // convert to UnsafeMutablePointer to pass into the cpp interface
    let baseAddress = buffer.baseAddress
    let result = UnsafeMutablePointer(mutating: baseAddress)
    if result == nil {
        fatalError("Couldn't convert image bitmap to type UnsafeMutablePointer")
    }
    return result!
}

// complete transform pipeline
func transformImage(image: UIImage, width: Int, height: Int) -> UnsafeMutablePointer<Float16> {
    let resizedImage = resizeImage(image: image, width: width, height: height)
    let result = getBitMapImage(image: resizedImage)
    return result
}
