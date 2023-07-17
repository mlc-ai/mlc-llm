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
