import { db } from './db';
import { productsTable, salesTable } from './schema';

export async function seed() {
    console.log('🌱 Seeding database...');

    // Insert products
    await db.insert(productsTable).values([
        { name: 'Laptop', category: 'Electronics', price: 999.99, stock: 50 },
        { name: 'Mouse', category: 'Electronics', price: 25.99, stock: 200 },
        { name: 'Keyboard', category: 'Electronics', price: 75.0, stock: 150 },
        { name: 'Monitor', category: 'Electronics', price: 299.99, stock: 75 },
        { name: 'Desk Chair', category: 'Furniture', price: 199.99, stock: 40 },
        { name: 'Desk', category: 'Furniture', price: 399.99, stock: 30 },
        { name: 'Notebook', category: 'Stationery', price: 5.99, stock: 500 },
        { name: 'Pen Set', category: 'Stationery', price: 12.99, stock: 300 },
    ]);

    // Insert sales
    await db.insert(salesTable).values([
        {
            product_id: 1,
            quantity: 2,
            total_amount: 1999.98,
            customer_name: 'John Doe',
            region: 'North',
        },
        {
            product_id: 2,
            quantity: 5,
            total_amount: 129.95,
            customer_name: 'Jane Smith',
            region: 'South',
        },
        {
            product_id: 3,
            quantity: 3,
            total_amount: 225.0,
            customer_name: 'Bob Johnson',
            region: 'East',
        },
        {
            product_id: 1,
            quantity: 1,
            total_amount: 999.99,
            customer_name: 'Alice Brown',
            region: 'West',
        },
        {
            product_id: 4,
            quantity: 2,
            total_amount: 599.98,
            customer_name: 'Charlie Wilson',
            region: 'North',
        },
        {
            product_id: 5,
            quantity: 4,
            total_amount: 799.96,
            customer_name: 'Diana Davis',
            region: 'South',
        },
        {
            product_id: 6,
            quantity: 1,
            total_amount: 399.99,
            customer_name: 'Eve Martinez',
            region: 'East',
        },
        {
            product_id: 7,
            quantity: 20,
            total_amount: 119.8,
            customer_name: 'Frank Lee',
            region: 'West',
        },
        {
            product_id: 8,
            quantity: 10,
            total_amount: 129.9,
            customer_name: 'Grace Kim',
            region: 'North',
        },
        {
            product_id: 2,
            quantity: 3,
            total_amount: 77.97,
            customer_name: 'Henry Chen',
            region: 'South',
        },
        {
            product_id: 3,
            quantity: 2,
            total_amount: 150.0,
            customer_name: 'Ivy Wang',
            region: 'East',
        },
        {
            product_id: 1,
            quantity: 1,
            total_amount: 999.99,
            customer_name: 'Jack Taylor',
            region: 'West',
        },
    ]);

    console.log('✅ Database seeded successfully!');
}

seed();