/**
* This code was generated by v0 by Vercel.
* @see https://v0.dev/t/TIu29xB3KWH
* Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
*/

/** Add fonts into your Next.js project:

import { Inter } from 'next/font/google'

inter({
  subsets: ['latin'],
  display: 'swap',
})

To read more about using these font, please visit the Next.js documentation:
- App Directory: https://nextjs.org/docs/app/building-your-application/optimizing/fonts
- Pages Directory: https://nextjs.org/docs/pages/building-your-application/optimizing/fonts
**/
import { Button } from "@/components/ui/button"
import { DialogTrigger, DialogTitle, DialogHeader, DialogContent, Dialog } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { useChat } from 'ai/react'
export function Chat() {
  
  const { messages, input, handleInputChange, handleSubmit,setMessages,append } = useChat()
  return (
    (<Dialog>
      <DialogTrigger asChild>
        <Button
          className="text-gray-800 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700"
          variant="outline">
          Open AI Chatbot
        </Button>
      </DialogTrigger>
      <DialogContent
        className="w-full max-w-xl h-full max-h-[600px] flex flex-col bg-white dark:bg-gray-900 shadow-xl rounded-xl">
        <DialogHeader
          className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <DialogTitle className="text-gray-900 dark:text-white font-bold">AI Chatbot</DialogTitle>
          <div>
            <Button
              className="text-gray-500 dark:text-gray-200 hover:text-gray-700 dark:hover:text-white"
              size="icon"
              title="Close the dialog"
              variant="ghost">
              <PanelTopCloseIcon className="h-5 w-5" />
            </Button>
          </div>
        </DialogHeader>
        <ScrollArea className="flex-1 overflow-y-auto">
          <div className="space-y-4 p-4">
            {messages.map((message, index) => (
              <>
              {message.role === "assistant" ? (
                <div
              className="group flex flex-col gap-2 py-2 border border-gray-200 border-gray-300 rounded-lg shadow-sm bg-white dark:border-gray-700 dark:bg-gray-800 dark:border-gray-800">
              <div
                className="flex-1 whitespace-pre-wrap p-2 text-sm prose prose-sm prose-neutral dark:text-white dark:prose-invert">
                <p>
                  <span className="font-semibold">AI:</span>
                  {message.content}
                </p>
                <p
                  className="text-right text-xs tracking-wide text-gray-800 dark:text-gray-300">
                  Apr 22, 2024 10:16 AM
                </p>
              </div>
            </div>

              ):(
                <div
              className="group flex flex-col gap-2 py-2 border border-gray-200 border-gray-300 rounded-lg shadow-sm bg-white dark:border-gray-700 dark:bg-gray-800 dark:border-gray-800">
              <div
                className="flex-1 whitespace-pre-wrap p-2 text-sm prose prose-sm prose-neutral dark:text-white dark:prose-invert">
                <p>
                  <span className="font-semibold">Alice:</span>
                  {message.content}
                </p>
                <p
                  className="text-right text-xs tracking-wide text-gray-800 dark:text-gray-300">
                  Apr 22, 2024 10:15 AM
                </p>
              </div>
            </div>
              )}
              </>
            ))}
            
            
          </div>
        </ScrollArea>
        <div
        
        
        
          className="flex items-center reversed:flex-row-reverse gap-2 p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <form onSubmit={handleSubmit}>
          <Input
            className="bending-transition"
            clearable
            placeholder="Ask me anything!"
            type="text" />
            onChange={handleInputChange}
          <Button
            className="text-scheme-accent bg-transparent hover:bg-gray-200 dark:hover:bg-gray-700"
            variant="outline">
            <PlusIcon className="h-5 w-5" />
            <span className="sr-only">Add attachment</span>
          </Button>
          <Button onClick={handleSubmit}
            className="text-white bg-blue-500 hover:bg-blue-600 dark:hover:bg-blue-700">
            <SendIcon className="h-5 w-5 mr-2" />
            <span>Send</span>
            
          </Button>
          </form>
        </div>
      </DialogContent>
    </Dialog>)
  );
}

function PanelTopCloseIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
      <line x1="3" x2="21" y1="9" y2="9" />
      <path d="m9 16 3-3 3 3" />
    </svg>)
  );
}


function PlusIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>)
  );
}


function SendIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path d="m22 2-7 20-4-9-9-4Z" />
      <path d="M22 2 11 13" />
    </svg>)
  );
}
