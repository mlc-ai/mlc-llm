//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//  LLM Chat Module
//
// Exposed interface of Object-C, enables swift binding.
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <os/proc.h>

/**
 * The chat module that can be used by the swift app.
 * It is a centralized interface that also provides multimodal support, i.e. vision modules.
 *
 * A chat flow can be implemented as follows, for each round of conversation
 *
 * @code
 *
 *   chat.prefill(input);
 *   while(!chat.stopped()) {
 *     displayReply(chat.getMessage());
 *     chat.decode();
 *   }
 *
 * @endcode
 *
 * The execution logic of this module should be placed on a dedicated thread.
 *
 * @seealso ThreadWorker
 */
@interface ChatModule : NSObject

/**
 * Unload the current model and free all memory.
 * @note This function is useful to get memory estimation before launch next model.
 */
- (void)unload;

/**
 * Reload the chat module to a new model.
 *
 * @param modelLib The name of the modelLib
 * @param modelPath The path to the model artifacts.
 * @param appConfigJson The partial config that is used to partially override the model
 * configuration.
 */
- (void)reload:(NSString*)modelLib
        modelPath:(NSString*)modelPath
    appConfigJson:(NSString*)appConfigJson;

/**
 * Reset the current chat session.
 */
- (void)resetChat;

/**
 * Run prefill stage for a given input and decode the first output token.
 *
 *@param input The user input prompt.
 */
- (void)prefill:(NSString*)input;

/**
 *Run one decode step to decode the next token.
 */
- (void)decode;

/**
 * @returns The output message in the current round.
 */
- (NSString*)getMessage;

/**
 * @returns Whether the current round stopped
 */
- (bool)stopped;

/**
 * Get the runtime statistics for the chat module, and optionally the image module.
 *
 *@param useVision Whether an image module is used.
 */
- (NSString*)runtimeStatsText:(bool)useVision;

/**
 * Pre-process by prefilling the system prompts, running prior to any user input.
 */
- (void)processSystemPrompts;

/**
 * \brief Run one round of prefill and decode.
 *
 *  This function is not supposed to be used by apps.
 *  and is only included here when setting up the app
 *  for debugging purposes.
 */
- (void)evaluate;

/**
 * Unload the current image model and free all memory.
 * @note This function is useful to get memory estimation before launch next model.
 */
- (void)unloadImageModule;

/**
 * Reload the image module to a new model.
 *
 * @param modelLib The name of the modelLib
 * @param modelPath The path to the model artifacts.
 */
- (void)reloadImageModule:(NSString*)modelLib modelPath:(NSString*)modelPath;

/**
 * Reset the current image model.
 */
- (void)resetImageModule;

/**
 * Prefill the LLM with the embedding of the input image.
 *
 * @param image The uploaded image.
 * @param prevPlaceholder The previous placeholder in the prompt, i.e. <Img>.
 * @param postPlaceholder The post placeholder in the prompt, i.e. </Img>.
 */
- (void)prefillImage:(UIImage*)image
     prevPlaceholder:(NSString*)prevPlaceholder
     postPlaceholder:(NSString*)postPlaceholder;
@end
