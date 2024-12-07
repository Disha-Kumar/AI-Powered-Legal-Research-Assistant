import pandas as pd

def collect_data():
    # Example data collection: Legal documents
    data = {
        'document_id': [1, 2, 3, 4, 5],
        'content': [
            'Case law content 1',
            'Statute content 2',
            'Legal precedent content 3',
            'Case law content 4',
            'Statute content 5'
        ],
        'type': ['case law', 'statute', 'precedent', 'case law', 'statute']
    }
    df = pd.DataFrame(data)
    return df

# Example usage
df = collect_data()
print(df)
