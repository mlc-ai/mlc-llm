//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//
// Exposed interface of Object-C, enables swift binding.
#import <Foundation/Foundation.h>
#include <os/proc.h>

/**
 * The chat module that can be used by  the swift app.
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
 */
- (void)reload:(NSString*)modelLib modelPath:(NSString*)modelPath;

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
 * @returns Runtime stats of last runs.
 */
- (NSString*)runtimeStatsText;

/**
 * \brief Run one round of prefill and decode.
 *
 *  This function is not supposed to be used by apps.
 *  and is only included here when setting up the app
 *  for debugging purposes.
 */
- (void)evaluate;
@end
