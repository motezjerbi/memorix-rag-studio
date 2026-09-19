import rag_core as core

vs = core.load_vectorstore()
q = "Comment fonctionne un algorithme de Random Forest ?"
for doc, score in core.retrieve_with_scores(vs, q)[:4]:
    print(f"\n--- distance {score:.3f} | {doc.metadata.get('source')}")
    print(doc.page_content[:250].replace("\n", " "))